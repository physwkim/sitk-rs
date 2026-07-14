// C++ (ITK) side of the sitk-rs benchmark harness.
// Contract: doc/bench-spec.md. Input generator, parameters and the NDJSON
// schema are fixed by that document; nothing here may deviate from it.

#include "itkImage.h"
#include "itkMultiThreaderBase.h"

#include "itkRescaleIntensityImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkMeanImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkFFTConvolutionImageFilter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using FloatImage = itk::Image<float, 3>;
using UCharImage = itk::Image<unsigned char, 3>;
using UIntImage = itk::Image<unsigned int, 3>;

// ---------------------------------------------------------------- generator

// doc/bench-spec.md §Inputs: xorshift64*, state = seed, advanced once per
// voxel, value = ((state * 0x2545F4914F6CDD1D) >> 33) % 1000.
static std::vector<float>
synth(std::uint64_t seed, const std::size_t size[3])
{
  const std::size_t n = size[0] * size[1] * size[2];
  std::vector<float> v(n);
  std::uint64_t state = seed;
  for (std::size_t i = 0; i < n; ++i)
  {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const std::uint64_t x = state * 0x2545F4914F6CDD1DULL; // wrapping
    v[i] = static_cast<float>((x >> 33) % 1000);
  }
  return v;
}

// FNV-1a 64 over the raw little-endian bytes of the buffer.
static std::uint64_t
fnv1a64(const void * data, std::size_t nbytes)
{
  const auto * p = static_cast<const unsigned char *>(data);
  std::uint64_t h = 14695981039346656037ULL;
  for (std::size_t i = 0; i < nbytes; ++i)
  {
    h ^= static_cast<std::uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

template <typename T>
static std::uint64_t
checksum(const std::vector<T> & v)
{
  return fnv1a64(v.data(), v.size() * sizeof(T));
}

static std::string
hex64(std::uint64_t h)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(h));
  return buf;
}

template <typename TImage>
static typename TImage::Pointer
makeImage(const std::vector<typename TImage::PixelType> & data, const std::size_t size[3])
{
  typename TImage::SizeType sz;
  sz[0] = size[0];
  sz[1] = size[1];
  sz[2] = size[2];
  typename TImage::IndexType idx{};
  typename TImage::RegionType region(idx, sz);

  auto img = TImage::New();
  img->SetRegions(region);
  img->SetSpacing(itk::MakeFilled<typename TImage::SpacingType>(1.0));
  img->SetOrigin(itk::MakeFilled<typename TImage::PointType>(0.0));
  img->SetDirection(TImage::DirectionType::GetIdentity());
  img->Allocate();
  std::memcpy(img->GetBufferPointer(), data.data(), data.size() * sizeof(typename TImage::PixelType));
  return img;
}

// ---------------------------------------------------------------- op runner

struct RunResult
{
  double        ms;
  std::uint64_t out_checksum;
};

// Every op rebuilds its filter from scratch on every sample. A fresh filter
// has no cached output and an unset pipeline MTime, so Update() is forced to
// execute GenerateData() in full; the input image is a source-less Image, so
// nothing upstream is re-generated inside the timed region.
using OpFn = std::function<RunResult()>;

template <typename TFilter>
static RunResult
timeFilter(TFilter * filter)
{
  const auto t0 = std::chrono::steady_clock::now();
  filter->Update();
  const auto t1 = std::chrono::steady_clock::now();

  auto *      out = filter->GetOutput();
  const auto  n = out->GetBufferedRegion().GetNumberOfPixels();
  using Pixel = typename std::remove_reference<decltype(*out)>::type::PixelType;
  const std::uint64_t sum = fnv1a64(out->GetBufferPointer(), n * sizeof(Pixel));

  return { std::chrono::duration<double, std::milli>(t1 - t0).count(), sum };
}

// ---------------------------------------------------------------- stats

struct Stats
{
  double mean, median, stddev;
};

static Stats
stats(std::vector<double> v)
{
  Stats s{ 0, 0, 0 };
  if (v.empty())
    return s;
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  s.median = (v.size() % 2 == 0) ? 0.5 * (v[m - 1] + v[m]) : v[m];
  double acc = 0.0;
  for (double x : v)
    acc += (x - s.mean) * (x - s.mean);
  s.stddev = (v.size() > 1) ? std::sqrt(acc / (v.size() - 1)) : 0.0;
  return s;
}

// ---------------------------------------------------------------- main

static const char * OPS[] = { "rescale_intensity",
                              "smoothing_recursive_gaussian",
                              "discrete_gaussian",
                              "median",
                              "mean",
                              "gradient_magnitude",
                              "gradient_magnitude_recursive_gaussian",
                              "binary_dilate",
                              "signed_maurer_distance_map",
                              "connected_component",
                              "otsu_threshold",
                              "fft_convolution" };

struct SizeSpec
{
  const char * name;
  std::size_t  dim;
};
static const SizeSpec SIZES[] = { { "small", 64 }, { "medium", 256 }, { "large", 512 } };

// doc/bench-spec.md: `large` is skipped for ops whose ITK implementation would
// take > 120 s. Enforced by timing the first (untimed, not-reported) warmup call
// and bailing out if it exceeds this budget.
static constexpr double LARGE_BUDGET_MS = 120000.0;

// Warm up until this much wall time has been spent in untimed calls, not for a
// fixed *count* of calls.
//
// This box ramps: after it has been idle, the first ~2 s of a 96-thread pass runs
// up to 70% slow and decays (measured in bench/results/harness-instability-result.md
// -- criterion's own per-sample times on the Rust side reach within 5% of steady at
// 1.63 s of measured work, ~2.1 s counting the warm-up before them). That is a
// property of the machine, not of the language, so the C++ harness pays it too.
//
// A single warmup call cannot cover it. At 64^3 an ITK op takes 0.6-37 ms, so one
// call is <2% of the ramp and *all ten* samples then land inside it -- the harness
// reports how cold the box was, not what ITK costs. The Rust harness had the same
// defect with a 500 ms warm-up and it inflated its own 64^3 numbers by up to 2.02x.
//
// 3 s covers the measured 2.1 s ramp with margin. It is the same number the Rust
// harness uses (`WARM_UP_MS` in benches/bench_ops.rs) and it is derived from the
// ramp, not tuned until an ITK number came out somewhere pleasant. Expensive cells
// (a 512^3 op costing seconds) still get exactly one warmup call, which is what the
// loop condition already gives them.
static constexpr double WARM_UP_MS = 3000.0;

int
main(int argc, char ** argv)
{
  std::uint64_t seed = 42;
  int           samples = 10;
  std::string   config = "tN";
  std::string   outPath;
  std::vector<std::string> onlyOps, onlySizes;

  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto              next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--seed")
      seed = std::stoull(next());
    else if (a == "--samples")
      samples = std::stoi(next());
    else if (a == "--config")
      config = next();
    else if (a == "--out")
      outPath = next();
    else if (a == "--op")
      onlyOps.push_back(next());
    else if (a == "--size")
      onlySizes.push_back(next());
    else
    {
      std::cerr << "unknown argument: " << a << "\n";
      return 2;
    }
  }

  if (config == "t1")
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);
  else if (config != "tN")
  {
    std::cerr << "config must be t1 or tN\n";
    return 2;
  }
  const unsigned threads = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();

  std::cerr << "config=" << config << " threads=" << threads << " threader="
            << itk::MultiThreaderBase::ThreaderTypeToString(itk::MultiThreaderBase::GetGlobalDefaultThreader())
            << " seed=" << seed << " samples=" << samples << "\n";

  std::ofstream file;
  std::ostream * out = &std::cout;
  if (!outPath.empty())
  {
    file.open(outPath);
    out = &file;
  }

  for (const SizeSpec & ss : SIZES)
  {
    if (!onlySizes.empty() && std::find(onlySizes.begin(), onlySizes.end(), ss.name) == onlySizes.end())
      continue;

    const std::size_t size[3] = { ss.dim, ss.dim, ss.dim };
    const std::size_t voxels = size[0] * size[1] * size[2];

    // Input generation is outside every timed region.
    const std::vector<float> base = synth(seed, size);

    // Binary/label inputs: threshold the same volume at >= 500.0.
    std::vector<unsigned char> maskU8(voxels);
    std::vector<float>         maskF32(voxels);
    for (std::size_t i = 0; i < voxels; ++i)
    {
      const bool fg = base[i] >= 500.0f;
      maskU8[i] = fg ? 1 : 0;
      maskF32[i] = fg ? 1.0f : 0.0f;
    }

    const std::uint64_t sumBase = checksum(base);
    const std::uint64_t sumMaskU8 = checksum(maskU8);
    const std::uint64_t sumMaskF32 = checksum(maskF32);
    std::cerr << "input checksums " << ss.name << " (" << ss.dim << "^3): base_f32=" << hex64(sumBase)
              << " mask_u8=" << hex64(sumMaskU8) << " mask_f32=" << hex64(sumMaskF32) << "\n";

    auto baseImg = makeImage<FloatImage>(base, size);
    auto maskU8Img = makeImage<UCharImage>(maskU8, size);
    auto maskF32Img = makeImage<FloatImage>(maskF32, size);

    // 7^3 normalized box kernel for fft_convolution.
    const std::size_t  kdim[3] = { 7, 7, 7 };
    std::vector<float> kdata(7 * 7 * 7, 1.0f / 343.0f);
    auto               kernelImg = makeImage<FloatImage>(kdata, kdim);

    for (const char * op : OPS)
    {
      const std::string opName = op;
      if (!onlyOps.empty() && std::find(onlyOps.begin(), onlyOps.end(), opName) == onlyOps.end())
        continue;

      std::uint64_t inSum = sumBase;
      OpFn          fn;

      if (opName == "rescale_intensity")
      {
        fn = [&] {
          auto f = itk::RescaleIntensityImageFilter<FloatImage, FloatImage>::New();
          f->SetInput(baseImg);
          f->SetOutputMinimum(0.0f);
          f->SetOutputMaximum(255.0f);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "smoothing_recursive_gaussian")
      {
        fn = [&] {
          using F = itk::SmoothingRecursiveGaussianImageFilter<FloatImage, FloatImage>;
          auto             f = F::New();
          F::SigmaArrayType sigma;
          sigma.Fill(2.0);
          f->SetInput(baseImg);
          f->SetSigmaArray(sigma);
          f->SetNormalizeAcrossScale(false);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "discrete_gaussian")
      {
        fn = [&] {
          using F = itk::DiscreteGaussianImageFilter<FloatImage, FloatImage>;
          auto         f = F::New();
          F::ArrayType variance;
          variance.Fill(4.0);
          F::ArrayType maxError;
          maxError.Fill(0.01);
          f->SetInput(baseImg);
          f->SetVariance(variance);
          f->SetMaximumError(maxError);
          f->SetMaximumKernelWidth(32);
          f->SetUseImageSpacing(true);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "median")
      {
        fn = [&] {
          using F = itk::MedianImageFilter<FloatImage, FloatImage>;
          auto          f = F::New();
          F::RadiusType r;
          r.Fill(2);
          f->SetInput(baseImg);
          f->SetRadius(r);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "mean")
      {
        fn = [&] {
          using F = itk::MeanImageFilter<FloatImage, FloatImage>;
          auto          f = F::New();
          F::RadiusType r;
          r.Fill(2);
          f->SetInput(baseImg);
          f->SetRadius(r);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "gradient_magnitude")
      {
        fn = [&] {
          auto f = itk::GradientMagnitudeImageFilter<FloatImage, FloatImage>::New();
          f->SetInput(baseImg);
          f->SetUseImageSpacing(true);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "gradient_magnitude_recursive_gaussian")
      {
        fn = [&] {
          auto f = itk::GradientMagnitudeRecursiveGaussianImageFilter<FloatImage, FloatImage>::New();
          f->SetInput(baseImg);
          f->SetSigma(2.0);
          f->SetNormalizeAcrossScale(false);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "binary_dilate")
      {
        inSum = sumMaskU8;
        fn = [&] {
          using Kernel = itk::BinaryBallStructuringElement<unsigned char, 3>;
          Kernel             ball;
          Kernel::SizeType   r;
          r.Fill(3);
          ball.SetRadius(r);
          ball.CreateStructuringElement();

          auto f = itk::BinaryDilateImageFilter<UCharImage, UCharImage, Kernel>::New();
          f->SetInput(maskU8Img);
          f->SetKernel(ball);
          f->SetForegroundValue(1);
          f->SetBackgroundValue(0);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "signed_maurer_distance_map")
      {
        inSum = sumMaskF32;
        fn = [&] {
          auto f = itk::SignedMaurerDistanceMapImageFilter<FloatImage, FloatImage>::New();
          f->SetInput(maskF32Img);
          f->SetInsideIsPositive(false);
          f->SetSquaredDistance(false);
          f->SetUseImageSpacing(true);
          f->SetBackgroundValue(0.0f);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "connected_component")
      {
        inSum = sumMaskU8;
        fn = [&] {
          auto f = itk::ConnectedComponentImageFilter<UCharImage, UIntImage>::New();
          f->SetInput(maskU8Img);
          f->SetFullyConnected(false);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "otsu_threshold")
      {
        fn = [&] {
          auto f = itk::OtsuThresholdImageFilter<FloatImage, UCharImage>::New();
          f->SetInput(baseImg);
          f->SetNumberOfHistogramBins(128);
          f->SetInsideValue(1);
          f->SetOutsideValue(0);
          return timeFilter(f.GetPointer());
        };
      }
      else if (opName == "fft_convolution")
      {
        fn = [&] {
          auto f = itk::FFTConvolutionImageFilter<FloatImage, FloatImage, FloatImage>::New();
          f->SetInput(baseImg);
          f->SetKernelImage(kernelImg);
          f->SetNormalize(true);
          return timeFilter(f.GetPointer());
        };
      }
      else
      {
        std::cerr << "no such op: " << opName << "\n";
        return 2;
      }

      auto emit = [&](const Stats * s, int n, std::uint64_t outSum, const char * skipped) {
        *out << "{\"harness\":\"cpp\",\"op\":\"" << opName << "\",\"size\":\"" << ss.name << "\",\"voxels\":" << voxels
             << ",\"config\":\"" << config << "\",\"threads\":" << threads << ",";
        if (s)
        {
          char buf[256];
          std::snprintf(buf,
                        sizeof(buf),
                        "\"ms_mean\":%.4f,\"ms_median\":%.4f,\"ms_stddev\":%.4f,\"samples\":%d,",
                        s->mean,
                        s->median,
                        s->stddev,
                        n);
          *out << buf;
        }
        else
        {
          *out << "\"ms_mean\":null,\"ms_median\":null,\"ms_stddev\":null,\"samples\":0,";
        }
        *out << "\"input_checksum\":\"" << hex64(inSum) << "\",\"output_checksum\":"
             << (s ? ("\"" + hex64(outSum) + "\"") : std::string("null")) << ",\"skipped\":"
             << (skipped ? ("\"" + std::string(skipped) + "\"") : std::string("null")) << "}\n";
        out->flush();
      };

      // Warmup: also the >120 s gate for `large`. Repeated until WARM_UP_MS of
      // wall time has been spent, so the box's ramp is inside the warm-up instead
      // of inside the ten reported samples.
      RunResult warm;
      try
      {
        const auto warmStart = std::chrono::steady_clock::now();
        do
        {
          warm = fn();
        } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warmStart).count() <
                 WARM_UP_MS);
      }
      catch (const itk::ExceptionObject & e)
      {
        std::cerr << "!! " << opName << " " << ss.name << " " << config << ": " << e.what() << "\n";
        emit(nullptr, 0, 0, "itk exception");
        continue;
      }

      if (std::string(ss.name) == "large" && warm.ms > LARGE_BUDGET_MS)
      {
        std::cerr << "-- skip " << opName << " large: warmup " << warm.ms << " ms > 120 s\n";
        emit(nullptr, 0, 0, "too slow");
        continue;
      }

      std::vector<double> ms;
      std::uint64_t       outSum = warm.out_checksum;
      bool                unstable = false;
      for (int i = 0; i < samples; ++i)
      {
        const RunResult r = fn();
        if (r.out_checksum != outSum)
          unstable = true;
        ms.push_back(r.ms);
        std::cerr << "   " << opName << " " << ss.name << " " << config << " sample " << i << ": " << r.ms << " ms\n";
      }
      if (unstable)
        std::cerr << "!! " << opName << " " << ss.name << ": output checksum varied across samples\n";

      const Stats s = stats(ms);
      emit(&s, static_cast<int>(ms.size()), outSum, nullptr);
    }
  }
  return 0;
}
