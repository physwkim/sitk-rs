//! Limited-memory BFGS with bounds (`itk::LBFGSBOptimizerv4`).
//!
//! A faithful port of the Byrd–Lu–Nocedal–Zhu bound-constrained limited-memory
//! quasi-Newton method — the algorithm ITK wraps from netlib (`setulb`), the
//! standard optimizer SimpleITK offers for **deformable** registration (BSpline /
//! displacement field), where the parameter count is large and plain gradient
//! descent converges slowly. Each variable may be unbounded, bounded below,
//! bounded above, or bounded on both sides.
//!
//! The method builds an implicit approximation of the inverse Hessian from the
//! last `m` gradient/step pairs (the *compact representation* of Byrd, Nocedal &
//! Schnabel), finds the generalized Cauchy point along the projected-gradient
//! path to identify the active bounds, minimizes the quadratic model over the
//! free variables, and takes a projected Moré–Thuente line search step. It stops
//! when the projected-gradient infinity norm falls to `pgtol`, when the relative
//! function reduction falls below `factr · εmach`, or at the iteration /
//! function-evaluation caps.
//!
//! ## Port structure
//!
//! The netlib reverse-communication driver (`setulb` returning a `task` string to
//! request each function/gradient evaluation) is restructured into a direct loop
//! that calls the caller's `eval` closure where netlib would return `task="FG"`.
//! The numerics are otherwise a line-for-line port of `mainlb` and its
//! subroutines (`cauchy`, `freev`, `formk`, `cmprlb`, `subsm`, `lnsrlb`,
//! `matupd`, `formt`, `bmv`, `active`, `projgr`, `hpsolb`, plus the LINPACK
//! primitives `dpofa`/`dtrsl`), with Fortran's 1-based, column-major matrices
//! translated to 0-based flat `Vec<f64>` indexed `m[row + col*ld]`.
//!
//! Bounds and stopping-criterion names follow ITK: `BoundSelection` per variable
//! is `0` unbounded, `1` lower only, `2` both, `3` upper only;
//! `CostFunctionConvergenceFactor` is `factr`; `GradientConvergenceTolerance` is
//! `pgtol`. Per ITK, this optimizer does **not** support parameter scales.

use crate::registration::optimizer::{OptimizerResult, StopReason};

/// dot product of two equal-length slices (LINPACK `ddot`, unit stride).
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// `y += a·x` over equal-length slices (LINPACK `daxpy`, unit stride).
fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    if a == 0.0 {
        return;
    }
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Cholesky factorization of a symmetric positive-definite matrix (LINPACK
/// `dpofa`). `a` is column-major with leading dimension `lda`; on success only
/// the upper triangle (including diagonal) is overwritten with `R` such that
/// `a = Rᵀ·R`. Returns `0` on success, or the order `k` (1-based) of the leading
/// minor that is not positive definite.
fn dpofa(a: &mut [f64], lda: usize, n: usize) -> i32 {
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..j {
            // t = a[k,j] − dot(col k[0..k], col j[0..k]); a[k,j] ← t / a[k,k]
            let dot: f64 = (0..k).map(|i| a[i + k * lda] * a[i + j * lda]).sum();
            let t = (a[k + j * lda] - dot) / a[k + k * lda];
            a[k + j * lda] = t;
            s += t * t;
        }
        s = a[j + j * lda] - s;
        if s <= 0.0 {
            return (j + 1) as i32; // leading minor of order j+1 not pos. def.
        }
        a[j + j * lda] = s.sqrt();
    }
    0
}

/// Triangular solve (LINPACK `dtrsl`). Solves `t·x = b` or `tᵀ·x = b` in place
/// on `b`, where `t` is column-major triangular with leading dimension `ldt`.
/// `job` selects the system exactly as LINPACK: `00` lower `t·x=b`, `01` upper
/// `t·x=b`, `10` lower `tᵀ·x=b`, `11` upper `tᵀ·x=b`. Returns `0` if nonsingular,
/// else the index (1-based) of the first zero diagonal.
fn dtrsl(t: &[f64], ldt: usize, n: usize, b: &mut [f64], job: i32) -> i32 {
    // Check for zero diagonal elements.
    for info in 1..=n {
        if t[(info - 1) + (info - 1) * ldt] == 0.0 {
            return info as i32;
        }
    }

    // Determine the task: case 1 lower t·x=b, 2 upper t·x=b, 3 lower tᵀ·x=b,
    // 4 upper tᵀ·x=b (matching LINPACK's `job` decode).
    let mut case = 1;
    if job % 10 != 0 {
        case = 2;
    }
    if (job % 100) / 10 != 0 {
        case += 2;
    }

    match case {
        1 => {
            // solve t·x=b, t lower triangular
            b[0] /= t[0];
            for j in 1..n {
                let temp = -b[j - 1];
                // b[j..n] += temp · t[j..n, j-1]
                for i in j..n {
                    b[i] += temp * t[i + (j - 1) * ldt];
                }
                b[j] /= t[j + j * ldt];
            }
        }
        2 => {
            // solve t·x=b, t upper triangular
            b[n - 1] /= t[(n - 1) + (n - 1) * ldt];
            for jj in 2..=n {
                let j = n - jj; // 0-based index n-jj+1 (1-based) → n-jj
                let temp = -b[j + 1];
                // b[0..=j] += temp · t[0..=j, j+1]
                for i in 0..=j {
                    b[i] += temp * t[i + (j + 1) * ldt];
                }
                b[j] /= t[j + j * ldt];
            }
        }
        3 => {
            // solve tᵀ·x=b, t lower triangular
            b[n - 1] /= t[(n - 1) + (n - 1) * ldt];
            for jj in 2..=n {
                let j = n - jj;
                // b[j] -= dot(t[j+1..n, j], b[j+1..n])
                let dot: f64 = (j + 1..n).map(|i| t[i + j * ldt] * b[i]).sum();
                b[j] -= dot;
                b[j] /= t[j + j * ldt];
            }
        }
        4 => {
            // solve tᵀ·x=b, t upper triangular
            b[0] /= t[0];
            for j in 1..n {
                // b[j] -= dot(t[0..j, j], b[0..j])
                let dot: f64 = (0..j).map(|i| t[i + j * ldt] * b[i]).sum();
                b[j] -= dot;
                b[j] /= t[j + j * ldt];
            }
        }
        _ => unreachable!(),
    }
    0
}

/// Project the initial `x` onto the feasible box and initialize `iwhere`
/// (netlib `active`). Sets, per variable: `iwhere = -1` if unbounded, `3` if
/// fixed (`l == u`), else `0`. Returns `(cnstnd, boxed)` — whether any bound is
/// active at all, and whether every variable is doubly bounded.
fn active(
    n: usize,
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    iwhere: &mut [i32],
) -> (bool, bool) {
    let mut cnstnd = false;
    let mut boxed = true;

    // Project the initial x onto the feasible set if necessary.
    for i in 0..n {
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= l[i] {
                if x[i] < l[i] {
                    x[i] = l[i];
                }
            } else if nbd[i] >= 2 && x[i] >= u[i] && x[i] > u[i] {
                x[i] = u[i];
            }
        }
    }

    // Initialize iwhere and assign cnstnd and boxed.
    for i in 0..n {
        if nbd[i] != 2 {
            boxed = false;
        }
        if nbd[i] == 0 {
            iwhere[i] = -1; // always free
        } else {
            cnstnd = true;
            if nbd[i] == 2 && u[i] - l[i] <= 0.0 {
                iwhere[i] = 3; // always fixed
            } else {
                iwhere[i] = 0;
            }
        }
    }

    (cnstnd, boxed)
}

/// Product of the `2col × 2col` middle matrix `M` of the compact L-BFGS formula
/// with a vector `v`, returned in `p` (netlib `bmv`). `sy` holds `S'Y`, `wt` the
/// Cholesky factor `J'` of `θS'S + LD⁻¹L'`, both `m × m` column-major. Returns
/// `0`, or nonzero if the triangular system is singular.
#[allow(clippy::too_many_arguments)]
fn bmv(m: usize, sy: &[f64], wt: &[f64], col: usize, v: &[f64], p: &mut [f64]) -> i32 {
    if col == 0 {
        return 0;
    }
    // PART I: solve J p2 = v2 + L D⁻¹ v1.
    p[col] = v[col];
    for i in 2..=col {
        let i2 = col + i; // 1-based p index
        let mut sum = 0.0;
        for k in 1..=(i - 1) {
            sum += sy[(i - 1) + (k - 1) * m] * v[k - 1] / sy[(k - 1) + (k - 1) * m];
        }
        p[i2 - 1] = v[i2 - 1] + sum;
    }
    // Solve the triangular system J p2 = rhs (upper, transpose): job = 11.
    let info = dtrsl(wt, m, col, &mut p[col..col + col], 11);
    if info != 0 {
        return info;
    }
    // solve D^(1/2) p1 = v1.
    for i in 0..col {
        p[i] = v[i] / sy[i + i * m].sqrt();
    }
    // PART II: solve Jᵀ p2 = p2 (upper, no transpose): job = 01.
    let info = dtrsl(wt, m, col, &mut p[col..col + col], 1);
    if info != 0 {
        return info;
    }
    // compute p1 = −D^(−1/2)(p1 − D^(−1/2) L' p2) = −D^(−1/2) p1 + D⁻¹ L' p2.
    for i in 0..col {
        p[i] = -p[i] / sy[i + i * m].sqrt();
    }
    for i in 1..=col {
        let mut sum = 0.0;
        for k in (i + 1)..=col {
            sum += sy[(k - 1) + (i - 1) * m] * p[col + k - 1] / sy[(i - 1) + (i - 1) * m];
        }
        p[i - 1] += sum;
    }
    0
}

/// Compute `r = −Z'B(xcp−xk) − Z'g` for the subspace minimization (netlib
/// `cmprlb`), using `wa[2m..] = W'(xcp−x)` left by [`cauchy`]. `index[0..nfree]`
/// holds the free variables' (0-based) indices. Returns `0`, or `−8` if the
/// middle-matrix solve is singular.
#[allow(clippy::too_many_arguments)]
fn cmprlb(
    n: usize,
    m: usize,
    x: &[f64],
    g: &[f64],
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    wt: &[f64],
    z: &[f64],
    r: &mut [f64],
    wa: &mut [f64],
    index: &[usize],
    theta: f64,
    col: usize,
    head: usize,
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    if !cnstnd && col > 0 {
        for i in 0..n {
            r[i] = -g[i];
        }
    } else {
        for i in 0..nfree {
            let k = index[i];
            r[i] = -theta * (z[k] - x[k]) - g[k];
        }
        // p = wa[0..2col] ← M · (v = wa[2m..2m+2col]); the ranges are disjoint.
        {
            let (left, right) = wa.split_at_mut(2 * m);
            let info = bmv(m, sy, wt, col, &right[0..2 * col], &mut left[0..2 * col]);
            if info != 0 {
                return -8;
            }
        }
        let mut pointr = head; // 1-based column pointer into ws/wy
        for j in 1..=col {
            let a1 = wa[j - 1];
            let a2 = theta * wa[col + j - 1];
            for i in 0..nfree {
                let k = index[i];
                r[i] += wy[k + (pointr - 1) * n] * a1 + ws[k + (pointr - 1) * n] * a2;
            }
            pointr = pointr % m + 1;
        }
    }
    0
}

/// Form the upper half of the positive-definite `T = θS'S + LD⁻¹L'` in `wt`
/// (`m × m` column-major) and Cholesky-factor it to `J·J'` with `J'` in the
/// upper triangle (netlib `formt`). Returns `0`, or `−3` if `T` is not positive
/// definite.
fn formt(m: usize, wt: &mut [f64], sy: &[f64], ss: &[f64], col: usize, theta: f64) -> i32 {
    for j in 0..col {
        wt[j * m] = theta * ss[j * m];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = i.min(j) - 1;
            let mut ddum = 0.0;
            for k in 1..=k1 {
                ddum += sy[(i - 1) + (k - 1) * m] * sy[(j - 1) + (k - 1) * m]
                    / sy[(k - 1) + (k - 1) * m];
            }
            wt[(i - 1) + (j - 1) * m] = ddum + theta * ss[(i - 1) + (j - 1) * m];
        }
    }
    if dpofa(wt, m, col) != 0 {
        return -3;
    }
    0
}

/// Count the variables entering/leaving the free set since the previous
/// iteration and rebuild the free/active index partition at the GCP (netlib
/// `freev`). `index[0..nfree]` ends holding the free variables' (0-based)
/// indices and `index[nfree..n]` the bound ones; `indx2` records the changed
/// variables. Returns `(nfree, nenter, ileave, wrk)` where `ileave`/`nenter` are
/// the 1-based leaving-start / entering-count `formk` consumes and `wrk` says
/// whether the `K` factorization must be (re)formed.
#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    nfree_in: usize,
    index: &mut [usize],
    indx2: &mut [usize],
    iwhere: &[i32],
    updatd: bool,
    cnstnd: bool,
    iter: usize,
) -> (usize, usize, usize, bool) {
    let mut nenter = 0usize;
    let mut ileave = n + 1; // 1-based sentinel

    if iter > 0 && cnstnd {
        // Variables leaving the free set (were free, now bound).
        for &k in index.iter().take(nfree_in) {
            if iwhere[k] > 0 {
                ileave -= 1;
                indx2[ileave - 1] = k;
            }
        }
        // Variables entering the free set (were bound, now free).
        for &k in index.iter().take(n).skip(nfree_in) {
            if iwhere[k] <= 0 {
                nenter += 1;
                indx2[nenter - 1] = k;
            }
        }
    }
    let wrk = ileave < n + 1 || nenter > 0 || updatd;

    // Find the index set of free and active variables at the GCP.
    let mut nfree = 0usize;
    let mut iact = n + 1; // 1-based
    for (i, &w) in iwhere.iter().enumerate().take(n) {
        if w <= 0 {
            index[nfree] = i;
            nfree += 1;
        } else {
            iact -= 1;
            index[iact - 1] = i;
        }
    }
    (nfree, nenter, ileave, wrk)
}

/// Extract the least element of `t` and heapify the rest (netlib `hpsolb`,
/// Williams' HEAPSORT). On exit `t[n-1]` holds the least element with its index
/// in `iorder[n-1]`, and `t[0..n-1]` is a heap. `iheap == 0` first builds the
/// heap from an unordered `t`.
fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [usize], iheap: i32) {
    if iheap == 0 {
        for k in 2..=n {
            let ddum = t[k - 1];
            let indxin = iorder[k - 1];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j - 1] {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }
    if n > 1 {
        let mut i = 1;
        let out = t[0];
        let indxou = iorder[0];
        let ddum = t[n - 1];
        let indxin = iorder[n - 1];
        loop {
            let mut j = i + i;
            if j < n {
                if t[j] < t[j - 1] {
                    j += 1;
                }
                if t[j - 1] < ddum {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;
        t[n - 1] = out;
        iorder[n - 1] = indxou;
    }
}

/// Update the limited-memory matrices with the newest `(s, y) = (d, r)` pair and
/// form the middle matrix in `B` (netlib `matupd`): append `d`/`r` as the newest
/// columns of `S`/`Y`, roll the oldest out when the memory is full, refresh the
/// upper triangle of `S'S` and lower triangle of `S'Y`, and set `θ = y'y / y's`.
#[allow(clippy::too_many_arguments)]
fn matupd(
    n: usize,
    m: usize,
    ws: &mut [f64],
    wy: &mut [f64],
    sy: &mut [f64],
    ss: &mut [f64],
    d: &[f64],
    r: &[f64],
    itail: &mut usize,
    iupdat: usize,
    col: &mut usize,
    head: &mut usize,
    theta: &mut f64,
    rr: f64,
    dr: f64,
    stp: f64,
    dtd: f64,
) {
    if iupdat <= m {
        *col = iupdat;
        *itail = (*head + iupdat - 2) % m + 1;
    } else {
        *itail = *itail % m + 1;
        *head = *head % m + 1;
    }
    // Store the new s = d and y = r as the itail-th columns of WS and WY.
    let it = *itail - 1;
    ws[it * n..it * n + n].copy_from_slice(d);
    wy[it * n..it * n + n].copy_from_slice(r);
    *theta = rr / dr;

    let c = *col;
    if iupdat > m {
        // Roll the oldest information out of the upper triangle of SS and the
        // lower triangle of SY.
        for j in 1..=(c - 1) {
            for rr_ in 0..j {
                ss[rr_ + (j - 1) * m] = ss[(rr_ + 1) + j * m];
            }
            for rr_ in 0..(c - j) {
                sy[(j - 1 + rr_) + (j - 1) * m] = sy[(j + rr_) + j * m];
            }
        }
    }
    // Add the new information: the last row of SY and last column of SS.
    let mut pointr = *head;
    for j in 1..=(c - 1) {
        sy[(c - 1) + (j - 1) * m] = ddot(d, &wy[(pointr - 1) * n..(pointr - 1) * n + n]);
        ss[(j - 1) + (c - 1) * m] = ddot(&ws[(pointr - 1) * n..(pointr - 1) * n + n], d);
        pointr = pointr % m + 1;
    }
    ss[(c - 1) + (c - 1) * m] = if stp == 1.0 { dtd } else { stp * stp * dtd };
    sy[(c - 1) + (c - 1) * m] = dr;
}

/// Form the `LEL'` factorization of the indefinite middle matrix `K` in `wn`
/// (netlib `formk`), incrementally maintaining the inner-product matrix `wn1`
/// (both `2m × 2m` column-major). `ind[0..nsub]` are the subspace (free) variable
/// indices; `indx2`/`nenter`/`ileave` describe the variables that entered/left
/// the free set since the previous iteration. Returns `0`, `−1` if the first
/// Cholesky factorization failed, or `−2` if the second failed.
#[allow(clippy::too_many_arguments)]
fn formk(
    n: usize,
    nsub: usize,
    ind: &[usize],
    nenter: usize,
    ileave: usize,
    indx2: &[usize],
    iupdat: usize,
    updatd: bool,
    wn: &mut [f64],
    wn1: &mut [f64],
    m: usize,
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
) -> i32 {
    let m2 = 2 * m;
    let upcl;
    if updatd {
        if iupdat > m {
            // Shift the old part of WN1.
            for jy in 1..=(m - 1) {
                let js = m + jy;
                for r in 0..(m - jy) {
                    wn1[(jy - 1 + r) + (jy - 1) * m2] = wn1[(jy + r) + jy * m2];
                }
                for r in 0..(m - jy) {
                    wn1[(js - 1 + r) + (js - 1) * m2] = wn1[(js + r) + js * m2];
                }
                for r in 0..(m - 1) {
                    wn1[(m + r) + (jy - 1) * m2] = wn1[(m + 1 + r) + jy * m2];
                }
            }
        }
        // Put new rows in blocks (1,1), (2,1) and (2,2).
        let pend = nsub;
        let dbegin = nsub + 1;
        let dend = n;
        let iy_row = col - 1; // 0-based row 'col'
        let is_row = m + col - 1; // 0-based row 'm+col'
        let mut ipntr = head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            for &k1 in ind.iter().take(pend) {
                temp1 += wy[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
            }
            for k in dbegin..=dend {
                let k1 = ind[k - 1];
                temp2 += ws[k1 + (ipntr - 1) * n] * ws[k1 + (jpntr - 1) * n];
                temp3 += ws[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
            }
            wn1[iy_row + (jy - 1) * m2] = temp1;
            wn1[is_row + (js - 1) * m2] = temp2;
            wn1[is_row + (jy - 1) * m2] = temp3;
            jpntr = jpntr % m + 1;
        }
        // Put the new column in block (2,1).
        let jy_col = col - 1; // 0-based column 'col'
        let mut jpntr = head + col - 1;
        if jpntr > m {
            jpntr -= m;
        }
        let mut ipntr = head;
        for i in 1..=col {
            let is = m + i;
            let mut temp3 = 0.0;
            for &k1 in ind.iter().take(pend) {
                temp3 += ws[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
            }
            ipntr = ipntr % m + 1;
            wn1[(is - 1) + jy_col * m2] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }

    // Modify the old parts in blocks (1,1) and (2,2) for the changed free set.
    let mut ipntr = head;
    for iy in 1..=upcl {
        let is = m + iy;
        let mut jpntr = head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            let mut temp4 = 0.0;
            for &k1 in indx2.iter().take(nenter) {
                temp1 += wy[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
                temp2 += ws[k1 + (ipntr - 1) * n] * ws[k1 + (jpntr - 1) * n];
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1];
                temp3 += wy[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
                temp4 += ws[k1 + (ipntr - 1) * n] * ws[k1 + (jpntr - 1) * n];
            }
            wn1[(iy - 1) + (jy - 1) * m2] += temp1 - temp3;
            wn1[(is - 1) + (js - 1) * m2] += -temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    // Modify the old parts in block (2,1).
    let mut ipntr = head;
    for is in (m + 1)..=(m + upcl) {
        let mut jpntr = head;
        for jy in 1..=upcl {
            let mut temp1 = 0.0;
            let mut temp3 = 0.0;
            for &k1 in indx2.iter().take(nenter) {
                temp1 += ws[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1];
                temp3 += ws[k1 + (ipntr - 1) * n] * wy[k1 + (jpntr - 1) * n];
            }
            if is <= jy + m {
                wn1[(is - 1) + (jy - 1) * m2] += temp1 - temp3;
            } else {
                wn1[(is - 1) + (jy - 1) * m2] += -temp1 + temp3;
            }
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }

    // Form the upper triangle of WN from WN1.
    for iy in 1..=col {
        let is = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            wn[(jy - 1) + (iy - 1) * m2] = wn1[(iy - 1) + (jy - 1) * m2] / theta;
            wn[(js - 1) + (is - 1) * m2] = wn1[(is1 - 1) + (js1 - 1) * m2] * theta;
        }
        for jy in 1..iy {
            wn[(jy - 1) + (is - 1) * m2] = -wn1[(is1 - 1) + (jy - 1) * m2];
        }
        for jy in iy..=col {
            wn[(jy - 1) + (is - 1) * m2] = wn1[(is1 - 1) + (jy - 1) * m2];
        }
        wn[(iy - 1) + (iy - 1) * m2] += sy[(iy - 1) + (iy - 1) * m];
    }

    // Cholesky factor the (1,1) block to get L' in its upper triangle.
    if dpofa(wn, m2, col) != 0 {
        return -1;
    }
    // Form L⁻¹(−L_a'+R_z') in the (1,2) block.
    let col2 = 2 * col;
    for js in (col + 1)..=col2 {
        let mut bcol: Vec<f64> = (0..col).map(|r| wn[r + (js - 1) * m2]).collect();
        dtrsl(wn, m2, col, &mut bcol, 11);
        for (r, &v) in bcol.iter().enumerate() {
            wn[r + (js - 1) * m2] = v;
        }
    }
    // Form S'AA'S·θ + (L⁻¹(−L_a'+R_z'))'(L⁻¹(−L_a'+R_z')) in the (2,2) block.
    for is in (col + 1)..=col2 {
        for js in is..=col2 {
            let dot: f64 = (0..col)
                .map(|r| wn[r + (is - 1) * m2] * wn[r + (js - 1) * m2])
                .sum();
            wn[(is - 1) + (js - 1) * m2] += dot;
        }
    }
    // Cholesky factor the (2,2) block.
    if dpofa(&mut wn[col + col * m2..], m2, col) != 0 {
        return -2;
    }
    0
}

/// Subspace minimization (netlib `subsm`, with the Morales–Nocedal 2011 direct
/// primal safeguard): compute the unconstrained Newton direction of the
/// quadratic over the free variables via `K⁻¹`, project onto the box, and — if a
/// bound is hit with a positive directional derivative — backtrack to the box.
/// `ind[0..nsub]` are the free variables' (0-based) indices; on entry `x` is the
/// Cauchy point and `d` the reduced gradient, on exit `x` is the subspace
/// minimizer and `d` the Newton direction. Returns `(iword, info)`: `iword = 1`
/// if a bound was encountered (backtracked), `0` if the solution is in the box;
/// `info` is nonzero if the `K` solve is singular.
#[allow(clippy::too_many_arguments)]
fn subsm(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &[usize],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    d: &mut [f64],
    xp: &mut [f64],
    ws: &[f64],
    wy: &[f64],
    theta: f64,
    xx: &[f64],
    gg: &[f64],
    col: usize,
    head: usize,
    wv: &mut [f64],
    wn: &[f64],
) -> (i32, i32) {
    if nsub == 0 {
        return (0, 0);
    }
    // Compute wv = W'Zd.
    let mut pointr = head;
    for i in 1..=col {
        let mut temp1 = 0.0;
        let mut temp2 = 0.0;
        for j in 1..=nsub {
            let k = ind[j - 1];
            temp1 += wy[k + (pointr - 1) * n] * d[j - 1];
            temp2 += ws[k + (pointr - 1) * n] * d[j - 1];
        }
        wv[i - 1] = temp1;
        wv[col + i - 1] = theta * temp2;
        pointr = pointr % m + 1;
    }
    // Compute wv := K⁻¹ wv via the two triangular solves of the LEL' factor.
    let m2 = 2 * m;
    let col2 = 2 * col;
    let info = dtrsl(wn, m2, col2, &mut wv[..col2], 11);
    if info != 0 {
        return (0, info);
    }
    for wvi in wv.iter_mut().take(col) {
        *wvi = -*wvi;
    }
    let info = dtrsl(wn, m2, col2, &mut wv[..col2], 1);
    if info != 0 {
        return (0, info);
    }
    // Compute d = (1/θ)d + (1/θ²) Z'W wv.
    let mut pointr = head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=nsub {
            let k = ind[i - 1];
            d[i - 1] = d[i - 1]
                + wy[k + (pointr - 1) * n] * wv[jy - 1] / theta
                + ws[k + (pointr - 1) * n] * wv[js - 1];
        }
        pointr = pointr % m + 1;
    }
    let inv_theta = 1.0 / theta;
    for di in d.iter_mut().take(nsub) {
        *di *= inv_theta;
    }

    // Try the projection; d is the Newton direction.
    let mut iword = 0;
    xp[..n].copy_from_slice(&x[..n]);
    for i in 1..=nsub {
        let k = ind[i - 1];
        let dk = d[i - 1];
        let xk = x[k];
        if nbd[k] != 0 {
            if nbd[k] == 1 {
                x[k] = l[k].max(xk + dk);
                if x[k] == l[k] {
                    iword = 1;
                }
            } else if nbd[k] == 2 {
                let xk2 = l[k].max(xk + dk);
                x[k] = u[k].min(xk2);
                if x[k] == l[k] || x[k] == u[k] {
                    iword = 1;
                }
            } else if nbd[k] == 3 {
                x[k] = u[k].min(xk + dk);
                if x[k] == u[k] {
                    iword = 1;
                }
            }
        } else {
            x[k] = xk + dk;
        }
    }

    // `alpha` is 1 on every path that does not backtrack, so those exits report
    // iword = 0 (the full subspace step was taken).
    let mut alpha = 1.0;
    if iword != 0 {
        // Check the sign of the directional derivative.
        let mut dd_p = 0.0;
        for i in 0..n {
            dd_p += (x[i] - xx[i]) * gg[i];
        }
        if dd_p > 0.0 {
            // Positive directional derivative: backtrack along d to the box.
            x[..n].copy_from_slice(&xp[..n]);
            let mut temp1 = alpha;
            let mut ibd = 0;
            for i in 1..=nsub {
                let k = ind[i - 1];
                let dk = d[i - 1];
                if nbd[k] != 0 {
                    if dk < 0.0 && nbd[k] <= 2 {
                        let temp2 = l[k] - x[k];
                        if temp2 >= 0.0 {
                            temp1 = 0.0;
                        } else if dk * alpha < temp2 {
                            temp1 = temp2 / dk;
                        }
                    } else if dk > 0.0 && nbd[k] >= 2 {
                        let temp2 = u[k] - x[k];
                        if temp2 <= 0.0 {
                            temp1 = 0.0;
                        } else if dk * alpha > temp2 {
                            temp1 = temp2 / dk;
                        }
                    }
                    if temp1 < alpha {
                        alpha = temp1;
                        ibd = i;
                    }
                }
            }
            if alpha < 1.0 {
                let dk = d[ibd - 1];
                let k = ind[ibd - 1];
                if dk > 0.0 {
                    x[k] = u[k];
                    d[ibd - 1] = 0.0;
                } else if dk < 0.0 {
                    x[k] = l[k];
                    d[ibd - 1] = 0.0;
                }
            }
            for i in 1..=nsub {
                let k = ind[i - 1];
                x[k] += alpha * d[i - 1];
            }
        }
    }
    iword = if alpha < 1.0 { 1 } else { 0 };
    (iword, 0)
}

/// Compute a safeguarded trial step and update the bracketing interval
/// (`stx`, `sty`) for the Moré–Thuente line search (MINPACK-2 `dcstep`). `stx`
/// holds the best step so far; when `brackt` is set the minimizer is bracketed
/// in `[min(stx,sty), max(stx,sty)]`. `fp`/`dp` are the function/derivative at
/// the current trial `stp` (read-only); the endpoints and `stp` are updated in
/// place.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        // First case: a higher function value — the minimum is bracketed.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp < *stx {
            gamma = -gamma;
        }
        let p = gamma - *dx + theta;
        let q = gamma - *dx + gamma + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2.0 * (*stp - *stx);
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        *brackt = true;
    } else if sgnd < 0.0 {
        // Second case: a lower function value and derivatives of opposite sign —
        // the minimum is bracketed.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = gamma - dp + theta;
        let q = gamma - dp + gamma + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        // Third case: a lower function value, same-sign derivatives, magnitude of
        // the derivative decreases.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        // gamma = 0 only when the cubic does not tend to infinity along the step.
        let mut gamma = s * (0.0f64.max((theta / s).powi(2) - (*dx / s) * (dp / s))).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = gamma - dp + theta;
        let q = gamma + (*dx - dp) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            // The cubic step if closer to stp than the secant step, else secant.
            let mut cand = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            if *stp > *stx {
                cand = cand.min(*stp + 0.66 * (*sty - *stp));
            } else {
                cand = cand.max(*stp + 0.66 * (*sty - *stp));
            }
            stpf = cand;
        } else {
            // The cubic step if farther from stp than the secant step, else secant.
            let cand = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            stpf = cand.min(stpmax).max(stpmin);
        }
    } else {
        // Fourth case: a lower function value, same-sign derivatives, magnitude of
        // the derivative does not decrease.
        if *brackt {
            let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
            let s = theta.abs().max(dy.abs()).max(dp.abs());
            let mut gamma = s * ((theta / s).powi(2) - (*dy / s) * (dp / s)).sqrt();
            if *stp > *sty {
                gamma = -gamma;
            }
            let p = gamma - dp + theta;
            let q = gamma - dp + gamma + *dy;
            let r = p / q;
            stpf = *stp + r * (*sty - *stp);
        } else if *stp > *stx {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }

    // Update the interval which contains a minimizer.
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

/// The task requested by [`Dcsrch::step`] on return.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LsTask {
    /// Evaluate `f`, `g` at the new `stp` and call again.
    Fg,
    /// The sufficient-decrease and curvature conditions are satisfied.
    Conv,
    /// Rounding errors prevent further progress; `stp` is the best point found.
    Warn,
    /// An input argument was invalid.
    Error,
}

/// Moré–Thuente line search (MINPACK-2 `dcsrch`) as a reverse-communication
/// state machine: each [`step`](Self::step) either requests another
/// function/derivative evaluation ([`LsTask::Fg`]) or reports success/failure.
/// The netlib `csave`/`isave`/`dsave` save area becomes this struct's fields.
struct Dcsrch {
    started: bool,
    brackt: bool,
    stage: i32,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl Dcsrch {
    fn new() -> Self {
        Self {
            started: false,
            brackt: false,
            stage: 1,
            ginit: 0.0,
            gtest: 0.0,
            gx: 0.0,
            gy: 0.0,
            finit: 0.0,
            fx: 0.0,
            fy: 0.0,
            stx: 0.0,
            sty: 0.0,
            stmin: 0.0,
            stmax: 0.0,
            width: 0.0,
            width1: 0.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        f: f64,
        g: f64,
        stp: &mut f64,
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
    ) -> LsTask {
        if !self.started {
            // Initialization block: check inputs, set up the interval, request
            // the first evaluation.
            if *stp < stpmin
                || *stp > stpmax
                || g >= 0.0
                || ftol < 0.0
                || gtol < 0.0
                || xtol < 0.0
                || stpmin < 0.0
                || stpmax < stpmin
            {
                return LsTask::Error;
            }
            self.started = true;
            self.brackt = false;
            self.stage = 1;
            self.finit = f;
            self.ginit = g;
            self.gtest = ftol * g;
            self.width = stpmax - stpmin;
            self.width1 = self.width / 0.5;
            self.stx = 0.0;
            self.fx = self.finit;
            self.gx = self.ginit;
            self.sty = 0.0;
            self.fy = self.finit;
            self.gy = self.ginit;
            self.stmin = 0.0;
            self.stmax = *stp + 4.0 * *stp;
            return LsTask::Fg;
        }

        // If psi(stp) <= 0 and f'(stp) >= 0, enter the second stage.
        let ftest = self.finit + *stp * self.gtest;
        if self.stage == 1 && f <= ftest && g >= 0.0 {
            self.stage = 2;
        }
        let mut task = LsTask::Fg;
        // Warnings.
        if self.brackt && (*stp <= self.stmin || *stp >= self.stmax) {
            task = LsTask::Warn;
        }
        if self.brackt && self.stmax - self.stmin <= xtol * self.stmax {
            task = LsTask::Warn;
        }
        if *stp == stpmax && f <= ftest && g <= self.gtest {
            task = LsTask::Warn;
        }
        if *stp == stpmin && (f > ftest || g >= self.gtest) {
            task = LsTask::Warn;
        }
        // Convergence.
        if f <= ftest && g.abs() <= gtol * (-self.ginit) {
            task = LsTask::Conv;
        }
        if task == LsTask::Warn || task == LsTask::Conv {
            return task;
        }

        // A modified function predicts the step during the first stage when a
        // lower value has been obtained but the decrease is not yet sufficient.
        if self.stage == 1 && f <= self.fx && f > ftest {
            let fm = f - *stp * self.gtest;
            let mut fxm = self.fx - self.stx * self.gtest;
            let mut fym = self.fy - self.sty * self.gtest;
            let gm = g - self.gtest;
            let mut gxm = self.gx - self.gtest;
            let mut gym = self.gy - self.gtest;
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                stp,
                fm,
                gm,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
            self.fx = fxm + self.stx * self.gtest;
            self.fy = fym + self.sty * self.gtest;
            self.gx = gxm + self.gtest;
            self.gy = gym + self.gtest;
        } else {
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                stp,
                f,
                g,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
        }

        // Decide whether a bisection step is needed.
        if self.brackt {
            if (self.sty - self.stx).abs() >= 0.66 * self.width1 {
                *stp = self.stx + 0.5 * (self.sty - self.stx);
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        // Set the min/max steps allowed for stp.
        if self.brackt {
            self.stmin = self.stx.min(self.sty);
            self.stmax = self.stx.max(self.sty);
        } else {
            self.stmin = *stp + 1.1 * (*stp - self.stx);
            self.stmax = *stp + 4.0 * (*stp - self.stx);
        }
        // Force the step within [stpmin, stpmax].
        *stp = (*stp).max(stpmin);
        *stp = (*stp).min(stpmax);
        // If no further progress is possible, revert to the best point.
        if self.brackt
            && (*stp <= self.stmin
                || *stp >= self.stmax
                || self.stmax - self.stmin <= xtol * self.stmax)
        {
            *stp = self.stx;
        }
        LsTask::Fg
    }
}

/// Best-position and evaluation-count bookkeeping mirroring the vnl driver's
/// `x_best` / `num_evaluations_` tracking: the returned solution is the
/// lowest-value point ever evaluated, not necessarily the last iterate.
struct Book {
    num_evaluations: usize,
    max_function_evaluations: usize,
    x_best: Vec<f64>,
    end_error: f64,
}

impl Book {
    /// Record an evaluation of value `f` at `x`; return `true` once the
    /// evaluation count exceeds the maximum (`num_evaluations > max`).
    fn record(&mut self, x: &[f64], f: f64) -> bool {
        if self.num_evaluations == 0 || f < self.end_error {
            self.end_error = f;
            self.x_best.copy_from_slice(x);
        }
        self.num_evaluations += 1;
        self.num_evaluations > self.max_function_evaluations
    }
}

// Line-search tolerances passed to dcsrch (netlib `lnsrlb` constants
// c_b280/c_b281/c_b282/c_b9): sufficient-decrease, curvature, interval, min step.
const LS_FTOL: f64 = 0.001;
const LS_GTOL: f64 = 0.9;
const LS_XTOL: f64 = 0.1;
const LS_STPMIN: f64 = 0.0;

/// The bound-constrained L-BFGS-B driver (netlib `mainlb` + the vnl reverse-
/// communication loop), restructured to call `eval` inline where netlib returns
/// `task = "FG"`. Returns the best point found, matching ITK's behavior.
#[allow(clippy::too_many_arguments)]
fn minimize<F>(
    n: usize,
    m: usize,
    x0: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    factr: f64,
    pgtol: f64,
    max_iterations: usize,
    max_function_evaluations: usize,
    mut eval: F,
) -> OptimizerResult
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let epsmch = f64::EPSILON;
    let tol = factr * epsmch;

    // Working matrices (column-major) and vectors.
    let mut ws = vec![0.0f64; n * m];
    let mut wy = vec![0.0f64; n * m];
    let mut sy = vec![0.0f64; m * m];
    let mut ss = vec![0.0f64; m * m];
    let mut wt = vec![0.0f64; m * m];
    let mut wn = vec![0.0f64; 4 * m * m];
    let mut snd = vec![0.0f64; 4 * m * m];
    let mut z = vec![0.0f64; n];
    let mut r = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut t = vec![0.0f64; n];
    let mut xp = vec![0.0f64; n];
    let mut wa = vec![0.0f64; 8 * m];
    let mut index = vec![0usize; n];
    let mut iwhere = vec![0i32; n];
    let mut indx2 = vec![0usize; n];

    // Scalars (netlib names).
    let mut col = 0usize;
    let mut head = 1usize;
    let mut theta = 1.0f64;
    let mut iupdat = 0usize;
    let mut updatd = false;
    let mut itail = 0usize;
    let mut nfree = n;
    let mut nenter = 0usize;
    let mut ileave = n + 1;
    let mut iter = 0usize;

    let mut x = x0.to_vec();
    let (cnstnd, boxed) = active(n, l, u, nbd, &mut x, &mut iwhere);

    let mut book = Book {
        num_evaluations: 0,
        max_function_evaluations,
        x_best: x.clone(),
        end_error: f64::INFINITY,
    };

    // Compute f0, g0.
    let (mut f, g0) = eval(&x);
    let mut g = g0;
    if book.record(&x, f) {
        return OptimizerResult {
            parameters: book.x_best,
            value: book.end_error,
            iterations: 0,
            stop_reason: StopReason::MaxFunctionEvaluations,
        };
    }
    let mut sbgnrm = projgr(n, l, u, nbd, &x, &g);
    if sbgnrm <= pgtol {
        return OptimizerResult {
            parameters: book.x_best,
            value: book.end_error,
            iterations: 0,
            stop_reason: StopReason::GradientConverged,
        };
    }

    let mut gd;
    let mut gdold = 0.0f64;
    let mut stp;
    let mut dnorm;
    let mut dtd;
    let mut stpmx;

    let stop_reason;
    'outer: loop {
        // Compute the generalized Cauchy point, or skip it when the problem is
        // unconstrained and a limited-memory matrix already exists.
        let wrk;
        if !cnstnd && col > 0 {
            z.copy_from_slice(&x);
            wrk = updatd;
        } else {
            let (_nseg, info) = cauchy(
                n,
                &x,
                l,
                u,
                nbd,
                &g,
                &mut indx2,
                &mut iwhere,
                &mut t,
                &mut d,
                &mut z,
                m,
                &wy,
                &ws,
                &sy,
                &wt,
                theta,
                col,
                head,
                &mut wa,
                epsmch,
                sbgnrm,
            );
            if info != 0 {
                // Singular triangular system: refresh the memory and restart.
                col = 0;
                head = 1;
                theta = 1.0;
                iupdat = 0;
                updatd = false;
                continue 'outer;
            }
            let (nf, ne, il, w) = freev(
                n, nfree, &mut index, &mut indx2, &iwhere, updatd, cnstnd, iter,
            );
            nfree = nf;
            nenter = ne;
            ileave = il;
            wrk = w;
        }

        // Subspace minimization (skipped when there are no free variables or the
        // matrix is B = θI).
        if nfree != 0 && col != 0 {
            let mut info = 0;
            if wrk {
                info = formk(
                    n, nfree, &index, nenter, ileave, &indx2, iupdat, updatd, &mut wn, &mut snd, m,
                    &ws, &wy, &sy, theta, col, head,
                );
            }
            if info == 0 {
                info = cmprlb(
                    n, m, &x, &g, &ws, &wy, &sy, &wt, &z, &mut r, &mut wa, &index, theta, col,
                    head, nfree, cnstnd,
                );
            }
            if info == 0 {
                let (_iword, i2) = subsm(
                    n, m, nfree, &index, l, u, nbd, &mut z, &mut r, &mut xp, &ws, &wy, theta, &x,
                    &g, col, head, &mut wa, &wn,
                );
                info = i2;
            }
            if info != 0 {
                // Singular system / bad factorization: refresh and restart.
                col = 0;
                head = 1;
                theta = 1.0;
                iupdat = 0;
                updatd = false;
                continue 'outer;
            }
        }

        // ---- Line search (netlib lnsrlb + dcsrch) ----
        for i in 0..n {
            d[i] = z[i] - x[i];
        }
        dtd = ddot(&d, &d);
        dnorm = dtd.sqrt();
        stpmx = 1e10;
        if cnstnd {
            if iter == 0 {
                stpmx = 1.0;
            } else {
                for i in 0..n {
                    let a1 = d[i];
                    if nbd[i] != 0 {
                        if a1 < 0.0 && nbd[i] <= 2 {
                            let a2 = l[i] - x[i];
                            if a2 >= 0.0 {
                                stpmx = 0.0;
                            } else if a1 * stpmx < a2 {
                                stpmx = a2 / a1;
                            }
                        } else if a1 > 0.0 && nbd[i] >= 2 {
                            let a2 = u[i] - x[i];
                            if a2 <= 0.0 {
                                stpmx = 0.0;
                            } else if a1 * stpmx > a2 {
                                stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
        }
        stp = if iter == 0 && !boxed {
            (1.0 / dnorm).min(stpmx)
        } else {
            1.0
        };
        t.copy_from_slice(&x);
        r.copy_from_slice(&g);
        let fold = f;
        let mut ifun = 0usize;
        let mut iback = 0usize;
        let mut ls = Dcsrch::new();
        let mut bad_direction = false;

        loop {
            gd = ddot(&g, &d);
            if ifun == 0 {
                gdold = gd;
                if gd >= 0.0 {
                    bad_direction = true;
                    break;
                }
            }
            let task_ls = ls.step(f, gd, &mut stp, LS_FTOL, LS_GTOL, LS_XTOL, LS_STPMIN, stpmx);
            if task_ls == LsTask::Conv || task_ls == LsTask::Warn {
                break; // NEW_X — line search succeeded
            }
            ifun += 1;
            iback = ifun - 1;
            if iback >= 20 {
                break; // too many backtracks
            }
            if stp == 1.0 {
                x.copy_from_slice(&z);
            } else {
                for i in 0..n {
                    x[i] = stp * d[i] + t[i];
                    if nbd[i] == 1 || nbd[i] == 2 {
                        x[i] = x[i].max(l[i]);
                    }
                    if nbd[i] == 2 || nbd[i] == 3 {
                        x[i] = x[i].min(u[i]);
                    }
                }
            }
            let (fv, gv) = eval(&x);
            f = fv;
            g.copy_from_slice(&gv);
            if book.record(&x, f) {
                stop_reason = StopReason::MaxFunctionEvaluations;
                break 'outer;
            }
        }

        if bad_direction || iback >= 20 {
            // Restore the previous iterate.
            x.copy_from_slice(&t);
            g.copy_from_slice(&r);
            f = fold;
            if col == 0 {
                stop_reason = StopReason::LineSearchFailed;
                break 'outer;
            }
            // Refresh the memory and restart the iteration.
            col = 0;
            head = 1;
            theta = 1.0;
            iupdat = 0;
            updatd = false;
            continue 'outer;
        }

        // ---- New iterate: optimality tests (netlib L777) ----
        iter += 1;
        sbgnrm = projgr(n, l, u, nbd, &x, &g);
        if iter >= max_iterations {
            stop_reason = StopReason::MaxIterations;
            break 'outer;
        }
        if sbgnrm <= pgtol {
            stop_reason = StopReason::GradientConverged;
            break 'outer;
        }
        let ddum = fold.abs().max(f.abs()).max(1.0);
        if fold - f <= tol * ddum {
            stop_reason = StopReason::Converged;
            break 'outer;
        }

        // Compute d = newx − oldx, r = newg − oldg, rr = y'y, dr = y's.
        for i in 0..n {
            r[i] = g[i] - r[i];
        }
        let rr = ddot(&r, &r);
        let dr;
        let ddum2;
        if stp == 1.0 {
            dr = gd - gdold;
            ddum2 = -gdold;
        } else {
            dr = (gd - gdold) * stp;
            for di in d.iter_mut() {
                *di *= stp;
            }
            ddum2 = -gdold * stp;
        }
        if dr <= epsmch * ddum2 {
            // Skip the L-BFGS update.
            updatd = false;
        } else {
            updatd = true;
            iupdat += 1;
            matupd(
                n, m, &mut ws, &mut wy, &mut sy, &mut ss, &d, &r, &mut itail, iupdat, &mut col,
                &mut head, &mut theta, rr, dr, stp, dtd,
            );
            if formt(m, &mut wt, &sy, &ss, col, theta) != 0 {
                // Nonpositive-definite T: refresh the memory and restart.
                col = 0;
                head = 1;
                theta = 1.0;
                iupdat = 0;
                updatd = false;
            }
        }
    }

    OptimizerResult {
        parameters: book.x_best,
        value: book.end_error,
        iterations: iter,
        stop_reason,
    }
}

/// Limited-memory BFGS optimizer with simple bound constraints
/// (`itk::LBFGSBOptimizerv4`).
///
/// Minimizes a scalar objective subject to per-variable bounds. Each variable's
/// bound type is set through [`set_bound_selection`](Self::set_bound_selection)
/// (`0` unbounded, `1` lower only, `2` both, `3` upper only), with the bound
/// values in [`set_lower_bound`](Self::set_lower_bound) /
/// [`set_upper_bound`](Self::set_upper_bound). With no bound selection every
/// variable is unbounded and the method reduces to unconstrained L-BFGS.
///
/// The returned [`OptimizerResult`] holds the **best** point found (lowest
/// objective value ever evaluated), matching ITK. Per ITK, this optimizer does
/// not support parameter scales.
#[derive(Clone, Debug)]
pub struct LBFGSBOptimizer {
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    bound_selection: Vec<i32>,
    cost_function_convergence_factor: f64,
    gradient_convergence_tolerance: f64,
    max_iterations: usize,
    max_function_evaluations: usize,
    max_corrections: usize,
}

impl LBFGSBOptimizer {
    /// An optimizer with the given iteration cap and ITK's defaults: no bounds
    /// (all variables unbounded), `CostFunctionConvergenceFactor = 1e7`,
    /// `GradientConvergenceTolerance = 1e-5`, `MaximumNumberOfFunctionEvaluations
    /// = 2000`, and `MaximumNumberOfCorrections = 5`.
    pub fn new(max_iterations: usize) -> Self {
        Self {
            lower_bound: Vec::new(),
            upper_bound: Vec::new(),
            bound_selection: Vec::new(),
            cost_function_convergence_factor: 1e7,
            gradient_convergence_tolerance: 1e-5,
            max_iterations,
            max_function_evaluations: 2000,
            max_corrections: 5,
        }
    }

    /// Set the per-variable lower bounds (length must equal the parameter count
    /// when a nonzero bound selection is used).
    pub fn set_lower_bound(&mut self, lower_bound: Vec<f64>) -> &mut Self {
        self.lower_bound = lower_bound;
        self
    }

    /// Set the per-variable upper bounds (length must equal the parameter count
    /// when a nonzero bound selection is used).
    pub fn set_upper_bound(&mut self, upper_bound: Vec<f64>) -> &mut Self {
        self.upper_bound = upper_bound;
        self
    }

    /// Set the per-variable bound type: `0` unbounded, `1` lower only, `2` both,
    /// `3` upper only (length must equal the parameter count).
    pub fn set_bound_selection(&mut self, bound_selection: Vec<i32>) -> &mut Self {
        self.bound_selection = bound_selection;
        self
    }

    /// Set the cost-function convergence factor `factr`: iteration stops when
    /// `(f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ factr · εmach`. Must be `≥ 0`.
    pub fn set_cost_function_convergence_factor(&mut self, factr: f64) -> &mut Self {
        self.cost_function_convergence_factor = factr;
        self
    }

    /// Set the projected-gradient tolerance `pgtol`: iteration stops when the
    /// infinity norm of the projected gradient falls to at or below `pgtol`.
    pub fn set_gradient_convergence_tolerance(&mut self, pgtol: f64) -> &mut Self {
        self.gradient_convergence_tolerance = pgtol;
        self
    }

    /// Set the maximum number of objective evaluations.
    pub fn set_max_function_evaluations(&mut self, max_function_evaluations: usize) -> &mut Self {
        self.max_function_evaluations = max_function_evaluations;
        self
    }

    /// Set the maximum number of stored variable-metric corrections `m` (the
    /// limited-memory depth).
    pub fn set_max_corrections(&mut self, max_corrections: usize) -> &mut Self {
        self.max_corrections = max_corrections;
        self
    }

    /// Minimize `eval` from `initial`, where `eval(p)` returns `(value,
    /// gradient)`. Returns the best point found.
    ///
    /// Panics on invalid input (mirroring netlib `errclb`): a zero `initial`,
    /// `max_corrections == 0`, a negative convergence factor, a `bound_selection`
    /// whose length or values are invalid, missing bound arrays, or an infeasible
    /// bound (`lower > upper` where both bounds apply).
    pub fn optimize<F>(&self, initial: Vec<f64>, eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let n = initial.len();
        assert!(n > 0, "initial parameters must be non-empty");
        assert!(self.max_corrections > 0, "max_corrections must be positive");
        assert!(
            self.cost_function_convergence_factor >= 0.0,
            "cost function convergence factor must be non-negative"
        );

        // Resolve the per-variable bound selection (empty ⇒ all unbounded).
        let nbd: Vec<i32> = if self.bound_selection.is_empty() {
            vec![0; n]
        } else {
            assert_eq!(
                self.bound_selection.len(),
                n,
                "bound selection length must equal parameter count"
            );
            self.bound_selection.clone()
        };
        for &b in &nbd {
            assert!((0..=3).contains(&b), "bound selection values must be 0..=3");
        }

        let needs_bounds = nbd.iter().any(|&b| b != 0);
        let (l, u): (Vec<f64>, Vec<f64>) = if needs_bounds {
            assert_eq!(
                self.lower_bound.len(),
                n,
                "lower bound length must equal parameter count"
            );
            assert_eq!(
                self.upper_bound.len(),
                n,
                "upper bound length must equal parameter count"
            );
            (self.lower_bound.clone(), self.upper_bound.clone())
        } else {
            (vec![0.0; n], vec![0.0; n])
        };
        for i in 0..n {
            if nbd[i] == 2 {
                assert!(
                    l[i] <= u[i],
                    "infeasible bound at index {i}: lower {} > upper {}",
                    l[i],
                    u[i]
                );
            }
        }

        minimize(
            n,
            self.max_corrections,
            &initial,
            &l,
            &u,
            &nbd,
            self.cost_function_convergence_factor,
            self.gradient_convergence_tolerance,
            self.max_iterations,
            self.max_function_evaluations,
            eval,
        )
    }
}

/// Infinity norm of the projected gradient (netlib `projgr`): for each variable,
/// clip the gradient component by the distance to its active bound, then take
/// the max absolute value. A NaN gradient component is propagated.
fn projgr(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.0f64;
    for i in 0..n {
        let mut gi = g[i];
        if gi.is_nan() {
            return gi;
        }
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    gi = (x[i] - u[i]).max(gi);
                }
            } else if nbd[i] <= 2 {
                gi = (x[i] - l[i]).min(gi);
            }
        }
        sbgnrm = sbgnrm.max(gi.abs());
    }
    sbgnrm
}

/// Compute the generalized Cauchy point (netlib `cauchy`): the first local
/// minimizer of the quadratic model `Q(x+s) = g's + ½s'Bs` along the projected
/// gradient path `P(x−tg, l, u)`. Returns the GCP in `xcp`, the Cauchy direction
/// in `d`, the free/fixed status in `iwhere`, and `W'(xcp−x)` in the `c` block of
/// `wa` (consumed by [`cmprlb`]). The work array `wa` (length `8m`) is split into
/// `p`, `c`, `wbp`, `v` blocks of `2m`. Returns `(nseg, info)` — segments
/// explored and `0`, or nonzero if the [`bmv`] solve is singular.
#[allow(clippy::too_many_arguments)]
fn cauchy(
    n: usize,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
    iorder: &mut [usize],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    m: usize,
    wy: &[f64],
    ws: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    wa: &mut [f64],
    epsmch: f64,
    sbgnrm: f64,
) -> (usize, i32) {
    let (p, rest) = wa.split_at_mut(2 * m);
    let (c, rest) = rest.split_at_mut(2 * m);
    let (wbp, v) = rest.split_at_mut(2 * m);

    if sbgnrm <= 0.0 {
        xcp[..n].copy_from_slice(&x[..n]);
        return (0, 0);
    }
    let mut bnded = true;
    let mut nfree = n + 1; // 1-based position for free-no-bound variables
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = 2 * col;
    let mut f1 = 0.0;
    for pi in p.iter_mut().take(col2) {
        *pi = 0.0;
    }

    // Determine each variable's bound status and breakpoint; build p = W'd.
    for i in 0..n {
        let neggi = -g[i];
        let mut tl = 0.0;
        let mut tu = 0.0;
        if iwhere[i] != 3 && iwhere[i] != -1 {
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.0;
            let xupper = nbd[i] >= 2 && tu <= 0.0;
            iwhere[i] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i] != 0 && iwhere[i] != -1 {
            d[i] = 0.0;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            for j in 0..col {
                p[j] += wy[i + (pointr - 1) * n] * neggi;
                p[col + j] += ws[i + (pointr - 1) * n] * neggi;
                pointr = pointr % m + 1;
            }
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.0 {
                // x(i) + d(i) is bounded below; compute its breakpoint.
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tl / (-neggi);
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else if nbd[i] >= 2 && neggi > 0.0 {
                // x(i) + d(i) is bounded above; compute its breakpoint.
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tu / neggi;
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else {
                // x(i) + d(i) is unbounded along the search direction.
                nfree -= 1;
                iorder[nfree - 1] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }

    if theta != 1.0 {
        for pj in p.iter_mut().skip(col).take(col) {
            *pj *= theta;
        }
    }
    xcp[..n].copy_from_slice(&x[..n]);
    if nbreak == 0 && nfree == n + 1 {
        return (0, 0);
    }

    for cj in c.iter_mut().take(col2) {
        *cj = 0.0;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let info = bmv(m, sy, wt, col, &p[..col2], &mut v[..col2]);
        if info != 0 {
            return (0, info);
        }
        f2 -= ddot(&v[..col2], &p[..col2]);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    let mut nseg = 1usize;

    let mut skip_move = false;
    if nbreak != 0 {
        let mut nleft = nbreak;
        let mut iter = 1usize;
        let mut tj = 0.0;
        loop {
            // Find the next smallest breakpoint.
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                tj = bkmin;
                ibp = iorder[ibkmin - 1];
            } else {
                if iter == 2 && ibkmin != nbreak {
                    // Replace the already-used smallest breakpoint with the last.
                    t[ibkmin - 1] = t[nbreak - 1];
                    iorder[ibkmin - 1] = iorder[nbreak - 1];
                }
                hpsolb(nleft, t, iorder, (iter - 2) as i32);
                tj = t[nleft - 1];
                ibp = iorder[nleft - 1];
            }
            let dt = tj - tj0;
            if dtm < dt {
                break; // a minimizer lies within this interval
            }
            // Otherwise fix variable ibp and zero its direction component.
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp];
            d[ibp] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = u[ibp] - x[ibp];
                xcp[ibp] = u[ibp];
                iwhere[ibp] = 2;
            } else {
                zibp = l[ibp] - x[ibp];
                xcp[ibp] = l[ibp];
                iwhere[ibp] = 1;
            }
            if nleft == 0 && nbreak == n {
                // All n variables are fixed; xcp is complete.
                dtm = dt;
                skip_move = true;
                break;
            }
            nseg += 1;
            let dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                daxpy(dt, &p[..col2], &mut c[..col2]);
                let mut pointr = head;
                for j in 0..col {
                    wbp[j] = wy[ibp + (pointr - 1) * n];
                    wbp[col + j] = theta * ws[ibp + (pointr - 1) * n];
                    pointr = pointr % m + 1;
                }
                let info = bmv(m, sy, wt, col, &wbp[..col2], &mut v[..col2]);
                if info != 0 {
                    return (nseg, info);
                }
                let wmc = ddot(&c[..col2], &v[..col2]);
                let wmp = ddot(&p[..col2], &v[..col2]);
                let wmw = ddot(&wbp[..col2], &v[..col2]);
                daxpy(-dibp, &wbp[..col2], &mut p[..col2]);
                f1 += dibp * wmc;
                f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw;
            }
            f2 = (epsmch * f2_org).max(f2);
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
            } else if bnded {
                // f1, f2 are set to zero in netlib here but are dead after the
                // loop (only dtm is used); the assignment is elided.
                dtm = 0.0;
            } else {
                dtm = -f1 / f2;
            }
            break;
        }
    }

    if !skip_move {
        if dtm <= 0.0 {
            dtm = 0.0;
        }
        tsum += dtm;
        // Move the free variables and those whose breakpoints were not reached.
        daxpy(tsum, &d[..n], &mut xcp[..n]);
    }
    // c = c + dtm·p = W'(xcp − x), used later by cmprlb.
    if col > 0 {
        daxpy(dtm, &p[..col2], &mut c[..col2]);
    }
    (nseg, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(p) = (p0 − 3)² + (p1 + 2)², unconstrained minimum at (3, −2).
    fn quadratic(p: &[f64]) -> (f64, Vec<f64>) {
        let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
        let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
        (v, g)
    }

    /// Rosenbrock f(p) = (1 − p0)² + 100(p1 − p0²)², minimum 0 at (1, 1).
    fn rosenbrock(p: &[f64]) -> (f64, Vec<f64>) {
        let v = (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
        let g = vec![
            -2.0 * (1.0 - p[0]) - 400.0 * p[0] * (p[1] - p[0] * p[0]),
            200.0 * (p[1] - p[0] * p[0]),
        ];
        (v, g)
    }

    #[test]
    fn unconstrained_quadratic_reaches_the_minimum() {
        let opt = LBFGSBOptimizer::new(200);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 3.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!(r.value < 1e-10, "value {}", r.value);
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
    }

    #[test]
    fn unconstrained_rosenbrock_converges() {
        // The curved Rosenbrock valley exercises the limited-memory BFGS updates
        // and the Moré–Thuente line search, not just a single Newton step.
        let opt = LBFGSBOptimizer::new(500);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert!((r.parameters[0] - 1.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8, "value {}", r.value);
    }

    #[test]
    fn active_lower_bound_pins_the_minimizer() {
        // The unconstrained minimum is at (3, −2); a lower bound l0 = 5 on p0
        // moves the constrained minimizer to p0 = 5 (active), p1 = −2.
        let mut opt = LBFGSBOptimizer::new(200);
        opt.set_bound_selection(vec![1, 0]) // p0 lower-bounded, p1 free
            .set_lower_bound(vec![5.0, 0.0])
            .set_upper_bound(vec![0.0, 0.0]);
        let r = opt.optimize(vec![10.0, 10.0], quadratic);
        assert!((r.parameters[0] - 5.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-6, "{:?}", r.parameters);
        // The constrained minimum value is (5−3)² = 4.
        assert!((r.value - 4.0).abs() < 1e-8, "value {}", r.value);
    }

    #[test]
    fn both_sided_bounds_are_respected_at_the_optimum() {
        // Box p ∈ [−1, 1]²; unconstrained minimum (3, −2) lies outside, so the
        // constrained minimizer is the nearest box corner (1, −1).
        let mut opt = LBFGSBOptimizer::new(200);
        opt.set_bound_selection(vec![2, 2])
            .set_lower_bound(vec![-1.0, -1.0])
            .set_upper_bound(vec![1.0, 1.0]);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 1.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!((r.parameters[1] + 1.0).abs() < 1e-6, "{:?}", r.parameters);
        for (i, &p) in r.parameters.iter().enumerate() {
            assert!((-1.0..=1.0).contains(&p), "param {i} = {p} out of box");
        }
    }

    #[test]
    fn bounded_rosenbrock_converges_to_the_constrained_corner() {
        // Rosenbrock over p ∈ [−2, 0.5]²: the free minimum (1, 1) is excluded, so
        // the constrained minimizer sits at p0 = 0.5 with p1 = p0² = 0.25.
        let mut opt = LBFGSBOptimizer::new(500);
        opt.set_bound_selection(vec![2, 2])
            .set_lower_bound(vec![-2.0, -2.0])
            .set_upper_bound(vec![0.5, 0.5]);
        let r = opt.optimize(vec![-1.0, -1.0], rosenbrock);
        assert!((r.parameters[0] - 0.5).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] - 0.25).abs() < 1e-3, "{:?}", r.parameters);
        for (i, &p) in r.parameters.iter().enumerate() {
            assert!((-2.0..=0.5).contains(&p), "param {i} = {p} out of box");
        }
    }

    #[test]
    fn max_function_evaluations_stops_and_returns_best() {
        // A tight evaluation budget stops early with the corresponding reason,
        // while still returning the best (lowest-value) point seen.
        let mut opt = LBFGSBOptimizer::new(10_000);
        opt.set_max_function_evaluations(3)
            .set_gradient_convergence_tolerance(0.0)
            .set_cost_function_convergence_factor(0.0);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert_eq!(r.stop_reason, StopReason::MaxFunctionEvaluations);
        // The returned value is the best seen, no worse than the start.
        let (start_value, _) = rosenbrock(&[-1.2, 1.0]);
        assert!(
            r.value <= start_value,
            "value {} start {}",
            r.value,
            start_value
        );
    }

    #[test]
    fn max_iterations_stops_with_that_reason() {
        let opt = LBFGSBOptimizer::new(2);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert_eq!(r.stop_reason, StopReason::MaxIterations);
        assert_eq!(r.iterations, 2);
    }

    #[test]
    fn already_at_the_minimum_converges_without_iterating() {
        let opt = LBFGSBOptimizer::new(100);
        let r = opt.optimize(vec![3.0, -2.0], quadratic);
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
        assert_eq!(r.iterations, 0);
        assert_eq!(r.parameters, vec![3.0, -2.0]);
    }

    #[test]
    fn high_dimensional_quadratic_with_limited_memory() {
        // n ≫ m exercises the compact limited-memory representation: a 20-D
        // separable quadratic with minimum at pᵢ = i, using only m = 5 corrections.
        let n = 20;
        let opt = LBFGSBOptimizer::new(500);
        let r = opt.optimize(vec![0.0; n], |p| {
            let mut v = 0.0;
            let mut g = vec![0.0; n];
            for i in 0..n {
                let d = p[i] - i as f64;
                v += d * d;
                g[i] = 2.0 * d;
            }
            (v, g)
        });
        for i in 0..n {
            assert!(
                (r.parameters[i] - i as f64).abs() < 1e-5,
                "param {i} = {}",
                r.parameters[i]
            );
        }
        assert!(r.value < 1e-8, "value {}", r.value);
    }

    #[test]
    #[should_panic(expected = "infeasible bound")]
    fn infeasible_bound_panics() {
        let mut opt = LBFGSBOptimizer::new(10);
        opt.set_bound_selection(vec![2])
            .set_lower_bound(vec![5.0])
            .set_upper_bound(vec![1.0]);
        opt.optimize(vec![0.0], |p| (p[0] * p[0], vec![2.0 * p[0]]));
    }
}
