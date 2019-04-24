use ndarray_linalg::*;
use ndarray::Array2;

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

    // Covariance matrix - not ideal right now.
    let transposed_mat1 = &mat1.t();
    let cov = transposed_mat1.dot(mat2);

    let (u, _s, v) = cov.svd(true, true).expect("Failed to perform SVD");

    // compute rotation matrix
    let rotation_matrix = u.unwrap().dot(&v.unwrap());

    rotation_matrix
}