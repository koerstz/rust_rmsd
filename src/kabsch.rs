use ndarray_linalg::*;
use ndarray::{Array,Array2};

pub fn rmsd(mat1: &Array2<f64>, mat2: &Array2<f64>) -> f64 {
    // Compute rmsd between the two matrices.

    let mut result: f64 = 0.0;
    for x in (mat1-mat2).outer_iter() {
        result += x.mapv(|a| a.powi(2)).scalar_sum();
    }

    let size: f64 = mat1.shape()[0] as f64;

    return (result/size).sqrt();
}

pub fn kabsch_rmsd(mat1: &Array2<f64>, mat2: &Array2<f64>) -> f64 {
    // Compute Kabsch rmsd.
     let rotated_mat1 = kabsch_rotate(mat1, mat2);
     rmsd(&rotated_mat1, mat2)
}

fn kabsch_rotate(mat1: &Array2<f64>, mat2: &Array2<f64>) -> Array2<f64> {
    let rotation_matrix = kabsch(mat1, mat2);
    let rotated_mat1 = mat1.dot(&rotation_matrix);

    rotated_mat1
}

fn kabsch(mat1: &Array2<f64>, mat2: &Array2<f64>) -> Array2<f64> {

    // Covariance matrix.
    let cov = mat1.t().dot(mat2);
    let usv = cov.svd(true, true).expect("Failed to perform SVD");

    let u = usv.0.unwrap();
    let v = usv.2.unwrap();

    // Correct our rotation matrix to ensure a right-handed coordinate system.
    let mut d = &v.det().unwrap() * &u.det().unwrap();
    if d > 0.0 {
        d = 1.0
    } else {
        d = -1.0
    }
    let mut m: Array2<f64> = Array::eye(3);
    m[[2,2]] = d;

    // Compute rotation matrix,
    let rotation_matrix = u.dot(&m).dot(&v);

    rotation_matrix
}