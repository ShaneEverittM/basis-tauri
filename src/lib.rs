use std::collections::VecDeque;
use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Display, Formatter};
use std::iter::{Iterator, Sum};
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Neg};

use anyhow::{anyhow, Error};
use num_traits::{One, Zero};

// RAYON

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    elements: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Copy> Matrix<T> {
    pub fn from_flat_vec(elements: Vec<T>, rows: usize, cols: usize) -> Result<Self, Error> {
        if rows * cols != elements.len() {
            return Err(anyhow!("Rows * columns must equal length of elements vec"));
        }

        let mut v: VecDeque<T> = elements.into();
        let mut acc: Vec<Vec<T>> = Vec::new();

        while v.len() >= cols {
            acc.push(v.drain(0..cols).collect());
            v.shrink_to_fit();
        }

        if !v.is_empty() {
            return Err(anyhow!("Must subdivide into rows"));
        }

        Ok(Self {
            elements: acc,
            rows,
            cols,
        })
    }

    pub fn from_array<const R: usize, const C: usize>(array: [[T; C]; R]) -> Self {
        let elements: Vec<Vec<T>> = Vec::from(array)
            .iter_mut()
            .map(|&mut row| Vec::from(row))
            .collect();

        Self {
            elements,
            rows: R,
            cols: C,
        }
    }

    pub fn from_vec(elements: Vec<Vec<T>>) -> Result<Self, Error> {
        let rows = elements.len();
        let cols = elements.first().unwrap().len();
        if elements.iter().any(|r| r.len() != cols) {
            Err(anyhow!("Bumpy matrix"))
        } else {
            Ok(Self {
                elements,
                rows,
                cols,
            })
        }
    }

    pub fn identity(n: usize) -> Self
        where
            T: One + Zero,
    {
        let mut res: Vec<T> = vec![Zero::zero(); n * n];

        let mut i;
        let mut j;
        for (idx, e) in res.iter_mut().enumerate() {
            i = Matrix::<T>::index_to_coordinates(idx, n).0;
            j = Matrix::<T>::index_to_coordinates(idx, n).1;
            if i == j {
                *e = One::one();
            }
        }
        Self::from_flat_vec(res, n, n).unwrap()
    }

    pub fn rows(&self) -> impl Iterator<Item = &Vec<T>> {
        self.elements.iter()
    }

    pub fn cols(&self) -> impl Iterator<Item = Box<dyn Iterator<Item = T> + '_>> + '_ {
        let mut iter_vec = Vec::new();

        for i in 0..self.cols {
            iter_vec.push(Box::new(
                self.elements
                    .iter()
                    .map(Box::new(move |row: &Vec<T>| row[i])),
            ) as Box<dyn Iterator<Item = T>>);
        }
        iter_vec.into_iter()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, Error>
        where
            T: Add<Output = T>,
    {
        self.element_wise_arithmetic_op(rhs, Add::add)
    }

    pub fn add_assign(&mut self, rhs: &Self) -> Result<(), Error>
        where
            T: AddAssign,
    {
        self.element_wise_update_op(rhs, AddAssign::add_assign)
    }

    pub fn scalar_add(&self, rhs: T) -> Self
        where
            T: Add<Output = T>,
    {
        self.scalar_op(rhs, Add::add)
    }

    pub fn scalar_add_assign(&mut self, rhs: T)
        where
            T: AddAssign,
    {
        self.scalar_update_op(rhs, AddAssign::add_assign)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, Error>
        where
            T: Sub<Output = T>,
    {
        self.element_wise_arithmetic_op(rhs, std::ops::Sub::sub)
    }

    pub fn sub_assign(&mut self, rhs: &Self) -> Result<(), Error>
        where
            T: SubAssign,
    {
        self.element_wise_update_op(rhs, SubAssign::sub_assign)
    }

    pub fn scalar_sub(&self, rhs: T) -> Self
        where
            T: Sub<Output = T>,
    {
        self.scalar_op(rhs, Sub::sub)
    }

    pub fn scalar_sub_assign(&mut self, rhs: T)
        where
            T: SubAssign,
    {
        self.scalar_update_op(rhs, SubAssign::sub_assign)
    }

    pub fn scalar_mul(&self, rhs: T) -> Self
        where
            T: Mul<Output = T>,
    {
        self.scalar_op(rhs, Mul::mul)
    }

    pub fn scalar_mul_assign(&mut self, rhs: T)
        where
            T: MulAssign,
    {
        self.scalar_update_op(rhs, MulAssign::mul_assign)
    }

    pub fn scalar_div(&self, rhs: T) -> Self
        where
            T: Div<Output = T>,
    {
        self.scalar_op(rhs, Div::div)
    }

    pub fn scalar_div_assign(&mut self, rhs: T)
        where
            T: DivAssign,
    {
        self.scalar_update_op(rhs, DivAssign::div_assign)
    }

    pub fn hadamard_product(&self, rhs: &Self) -> Result<Self, Error>
        where
            T: Mul<Output = T>,
    {
        self.element_wise_arithmetic_op(rhs, Mul::mul)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, Error>
        where
            T: Mul<Output = T> + Sum,
    {
        let mut result = Vec::new();
        for row in self.rows() {
            let mut row_result: Vec<T> = Vec::new();
            for col in rhs.cols() {
                let new_element = row.iter().zip(col.into_iter()).map(|(&x, y)| x * y).sum();
                row_result.push(new_element);
            }
            result.push(row_result)
        }
        Matrix::try_from(result)
    }

    pub fn mul_assign(&mut self, rhs: &Self) -> Result<(), Error>
        where
            T: Mul<Output = T> + Sum,
    {
        // Have to create a copy here because in place multiplication is impossible
        *self = self.clone().mul(&rhs)?;
        Ok(())
    }

    pub fn transpose(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j < i {
                    // SAFETY: x and y are aligned, non-null and non-overlapping.
                    // The swap will incur no re-allocations or moves so Vec will not be
                    // disturbed.
                    unsafe {
                        std::ptr::swap_nonoverlapping(
                            &mut self[i][j] as *mut T,
                            &mut self[j][i] as *mut T,
                            1,
                        );
                    }
                }
            }
        }
    }

    pub fn minor(&mut self) -> T
        where
            T: Mul<Output = T> + Sub<Output = T>
    {
        // This function exists to find the minor which is a component of a 2x2 matrix
        // Should the dimensions be checked?

        let a1 = self[0][0];
        let a2 = self[0][1];
        let a3 = self[1][0];
        let a4 = self[1][1];

        (a1 * a4) - (a2 * a3)
    }

    pub fn get_sub_matrix(&self, curr_col: usize, curr_row: usize) -> Self {
        let mut v: Vec<Vec<T>> = Vec::new();
        for i in 0..self.cols {
            let mut temp: Vec<T> = Vec::with_capacity(self.cols - 1);
            for j in 0..self.rows {
                if i != curr_col && j != curr_row {
                    temp.push(self[j][i]);
                }
            }
            if !(temp.is_empty()) {
                v.push(temp);
            }
        }

        Matrix::from_vec(v).unwrap()
    }

    pub fn determinant(&mut self) -> T
        where
            T: Mul<Output = T>,
            T: Sub<Output = T>,
            T: AddAssign,
            T: MulAssign,
            T: Neg<Output = T>,
            T: Zero
    {
        let mut sum: T = Zero::zero();

        // If a 2x2 matrix just return the determinant
        if self.rows == 2 && self.cols == 2 {
            return self.minor();
        }

        // We will use row 1 for calculating the determinant (i.e. i stays as 0)
        for j in 0..self.cols {
            // Extract the sub-matrix not containing row i and column j
            let mut sub_matrix = self.get_sub_matrix(j, 0);
            if sub_matrix.rows == 2 && sub_matrix.cols == 2 {
                // Determinant of the 2x2 sub-matrix
                let minor = sub_matrix.minor();
                let mut cofactor = self[0][j] * minor;
                // -1 ^ (i + j)
                if j % 2 == 1 {
                    cofactor = cofactor.neg();
                }
                sum += cofactor;
            } else {
                // Recursive call to work towards finding the determinant of the 2x2 sub-matrices
                let mut cofactor = self[0][j] * sub_matrix.determinant();
                // -1 ^ (i + j)
                if j % 2 == 1 {
                    cofactor = cofactor.neg();
                }
                sum += cofactor;
            }
        }

        sum
    }

    // For 2x2 and 3x3 matrices only
    pub fn cofactor_matrix_simple(&mut self) -> Result<Self, Error>
        where
            T: Mul<Output = T>,
            T: Sub<Output = T>,
            T: AddAssign,
            T: MulAssign,
            T: Neg<Output = T>,
            T: Zero
    {
        let mut cofactor_matrix: Vec<Vec<T>> = Vec::new();

        // If a 2x2 matrix just return the default cofactor matrix
        if self.rows == 2 && self.cols == 2 {
            let temp: Vec<T> = vec![(self[1][1]), self[0][1].neg()];
            cofactor_matrix.push(temp);
            let temp2: Vec<T> = vec![self[1][0].neg(), self[0][0]];
            cofactor_matrix.push(temp2);
        }

        // If a 3x3 matrix we cannot use the same determinant loop as the function
        else {
            for i in 0..self.cols {
                let mut temp: Vec<T> = Vec::new();
                for j in 0..self.rows {
                    let mut sub_matrix = self.get_sub_matrix(i, j);
                    // Determinant of the 2x2 sub-matrix
                    let mut minor = sub_matrix.minor();
                    // -1 ^ (i + j)
                    if (i + j) % 2 == 1 {
                        minor = minor.neg();
                    }
                    temp.push(minor);
                }
                cofactor_matrix.push(temp);
            }
        }
        Matrix::from_vec(cofactor_matrix)
    }

    pub fn cofactor_matrix_complex(&mut self) -> Result<Self, Error>
        where
            T: Mul<Output = T>,
            T: Sub<Output = T>,
            T: AddAssign,
            T: MulAssign,
            T: Neg<Output = T>,
            T: Zero
    {
        let mut cofactor_matrix: Vec<Vec<T>> = Vec::new();

        // For anything greater than 3x3
        for i in 0..self.cols {
            let mut temp: Vec<T> = Vec::new();
            for j in 0..self.rows {
                // Extract the sub-matrix not containing row i and column j
                let mut sub_matrix = self.get_sub_matrix(i, j);
                let mut minor = sub_matrix.determinant();
                if (i + j) % 2 == 1 {
                    minor = minor.neg();
                }
                temp.push(minor);
            }
            cofactor_matrix.push(temp);
        }

        Matrix::from_vec(cofactor_matrix)
    }


    pub fn invert(&mut self) -> Result<Self, Error>
        where
            T: Mul<Output = T>,
            T: Div<Output = T>,
            T: DivAssign,
            T: Sub<Output = T>,
            T: AddAssign,
            T: MulAssign,
            T: Neg<Output = T>,
            T: Zero + One

    {
        // First get the determinant
        let determinant = self.determinant();

        // Then get the cofactor matrix
        let mut inverse_matrix;
        if (self.rows == 2 && self.cols == 2) || (self.rows == 3 && self.cols == 3) {
            inverse_matrix = self.cofactor_matrix_simple()?;
        } else {
            inverse_matrix = self.cofactor_matrix_complex()?;
        }

        // If it is not a 2x2, transpose into the adjoined matrix
        // if self.rows != 2 && self.cols != 2 {
        //    inverse_matrix.transpose();
        //}

        // Scalar multiply by the reciprocal of the determinant
        //let numerator: T = One::one();
        //let scalar: T = numerator / determinant;
        inverse_matrix.scalar_div_assign(determinant);
        Ok(inverse_matrix)
    }

    fn element_wise_arithmetic_op(
        &self,
        rhs: &Self,
        op: impl Fn(T, T) -> T,
    ) -> Result<Self, Error> {
        self.dims_match(&rhs)?;

        self.iter()
            .zip(rhs.iter())
            .map(|(row1, row2)| {
                row1.iter()
                    .zip(row2.iter())
                    .map(|(&x, &y)| op(x, y))
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .try_into()
    }

    fn element_wise_update_op(
        &mut self,
        rhs: &Self,
        mut op: impl FnMut(&mut T, T),
    ) -> Result<(), Error> {
        self.dims_match(&rhs)?;

        self.iter_mut()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(x, &y)| op(x, y));

        Ok(())
    }

    fn scalar_op(&self, rhs: T, op: impl Fn(T, T) -> T) -> Self {
        self.iter()
            .map(|row| row.iter().map(|&e| op(e, rhs)).collect::<Vec<T>>())
            .collect::<Vec<Vec<T>>>()
            .try_into()
            .expect("Matrix is bumpy :^(")
    }

    fn scalar_update_op(&mut self, rhs: T, mut op: impl FnMut(&mut T, T)) {
        self.iter_mut().flatten().for_each(|e| op(e, rhs))
    }

    fn dims_match(&self, other: &Self) -> Result<(usize, usize), Error> {
        if self.rows == other.rows && self.cols == other.cols {
            Ok((self.rows, self.cols))
        } else {
            Err(anyhow!("Dimension mismatch"))
        }
    }

    fn index_to_coordinates(idx: usize, row_length: usize) -> (usize, usize) {
        let i = idx / row_length;
        let j = idx % row_length;
        (i, j)
    }
}

impl<T: Copy + Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in self.iter() {
            for e in row.iter() {
                f.write_fmt(format_args!("{}, ", e))?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl<T: Copy> Eq for Matrix<T> where T: Eq {}

impl<T: Copy> Deref for Matrix<T> {
    type Target = Vec<Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.elements
    }
}

impl<T: Copy> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.elements
    }
}

impl<T: Copy, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(array: [[T; C]; R]) -> Self {
        Self::from_array(array)
    }
}

impl<T: Copy> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Self::from_vec(value)
    }
}

impl<T: Copy> From<Matrix<T>> for Vec<Vec<T>> {
    fn from(m: Matrix<T>) -> Self {
        m.elements
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    macro_rules! ok {
        ($m:expr) => {{
            let m = $m;
            assert!(m.is_ok());
            m.unwrap()
        }};
    }

    #[test]
    fn addition() {
        let mut A = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let B = Matrix::from([[8, 7, 6, 5], [4, 3, 2, 1]]);

        let C = Matrix::from([[9, 9, 9, 9], [9, 9, 9, 9]]);

        assert_eq!(C, ok!(A.add(&B)));

        ok!(A.add_assign(&B));

        assert_eq!(A, C);

        let scalar = 1;

        let E = Matrix::from([[10, 10, 10, 10], [10, 10, 10, 10]]);

        let D = C.scalar_add(scalar);

        assert_eq!(D, E);
    }

    #[test]
    fn subtraction() {
        let mut A = Matrix::from([[5, 5], [5, 5]]);
        let B = Matrix::from([[1, 2], [3, 4]]);

        let C = ok!(A.sub(&B));

        assert_eq!(C, [[4, 3], [2, 1]].into());

        ok!(A.sub_assign(&B));

        assert_eq!(A, C);

        A.scalar_sub_assign(1);

        assert_eq!(A, [[3, 2], [1, 0]].into());
    }

    #[test]
    fn identity() {
        let identity = Matrix::identity(5);
        let manual_identity = Matrix::from([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]);

        assert_eq!(manual_identity, identity);
    }


    #[test]
    fn multiply() {
        let A = Matrix::from([[1, 2], [3, 4]]);
        let B = Matrix::from([[1, 3], [2, 4]]);

        let C = ok!(A.mul(&B));

        let D: Matrix<i32> = Matrix::from([[5, 11], [11, 25]]);

        assert_eq!(C, D);
    }

    #[test]
    fn from_vector() {
        let m1 = ok!(Matrix::try_from(vec![vec![1, 2], vec![3, 4]]));
        let m2 = Matrix::from([[1, 2], [3, 4]]);

        assert_eq!(m1, m2);
    }

    #[test]
    fn transpose() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let m2 = Matrix::from([[1, 3], [2, 4]]);

        m1.transpose();

        assert_eq!(m1, m2);
    }

    #[test]
    fn minor() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let check = -2;

        let res = m1.minor();

        assert_eq!(res, check);
    }

    #[test]
    fn determinant() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let check = -2;

        let res = m1.determinant();

        assert_eq!(res, check);
    }

    #[test]
    fn cofactor_2x2() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let m2 = Matrix::from([[4, -2], [-3, 1]]);

        let res = ok!(m1.cofactor_matrix_simple());

        assert_eq!(res, m2);
    }

    #[test]
    fn cofactor_3x3() {
        let mut m1 = Matrix::from([[1, 2, 3], [3, 1, 2], [4, 3, 1]]);
        let m2 = Matrix::from([[-5, 7, 1], [5, -11, 7], [5, 5, -5]]);

        let res = ok!(m1.cofactor_matrix_simple());

        assert_eq!(res, m2);
    }

    #[test]
    fn cofactor_4x4() {
        let mut m1 = Matrix::from([[1, 2, 3, 4], [3, 1, 2, 4], [4, 2, 1, 3], [2, 4, 3, 1]]);
        let m2 = Matrix::from([[20, -22, 5, -7], [-20, 38, -25, 3],
            [20, -42, 35, -17], [-20, 18, -15, 13]]);

        let res = ok!(m1.cofactor_matrix_complex());

        assert_eq!(res, m2);
    }

    #[test]
    fn inverse_2x2() {
        let mut m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix::from([[-2.0, 1.0], [1.5, -0.5]]);

        let res = m1.invert();

        assert_eq!(res, m2);
    }

    #[test]
    fn inverse_3x3() {
        let mut m1 = Matrix::from([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [4.0, 3.0, 1.0]]);
        let m2 = Matrix::from([[-0.25, 0.35, 0.05], [0.25, -0.55, 0.35], [0.25, 0.25, -0.25]]);

        let res = m1.invert();

        assert_eq!(res, m2);
    }

    #[test]
    fn inverse_4x4() {
        let mut m1 = Matrix::from([[1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 2.0, 4.0],
            [4.0, 2.0, 1.0, 3.0], [2.0, 4.0, 3.0, 1.0]]);
        let m2 = Matrix::from([[-0.5, 0.55, -0.125, 0.175], [0.5, -0.95, 0.625, -0.075],
            [-0.5, 1.05, -0.875, 0.425], [0.5, -0.45, 0.375, -0.325]]);

        let res = m1.invert();

        assert_eq!(res, m2);
    }
}