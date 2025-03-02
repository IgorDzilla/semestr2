from typing import Optional
from copy import deepcopy, copy


class Matrix:
    def __init__(
        self,
        dimensions: Optional[tuple[int, int]] = None,
        source: Optional[list[list]] = None,
    ) -> None:
        if source:
            self.__dimensions: tuple[int, int] = len(source), len(source[0])
            self.__matrix = [row[:] for row in source]
        elif dimensions:
            self.__dimensions: tuple[int, int] = dimensions
            self.__matrix = [[0] * dimensions[1] for _ in range(dimensions[0])]
        else:
            raise ValueError(
                "Can not initialize Matrix with neither dimensions, nor source."
            )

    def get_matrix(self) -> list[list]:
        return [row[:] for row in self.__matrix]

    def __getitem__(self, indexes: tuple[int, int]):
        return self.__matrix[indexes[0]][indexes[1]]

    def __iter__(self):
        for row in self.__matrix:
            for elem in row:
                yield elem

    def __setitem__(self, indexes: tuple[int, int], value) -> None:
        self.__matrix[indexes[0]][indexes[1]] = value

    def __str__(self) -> str:
        output: str = ""
        for row in self.__matrix:
            for el in row:
                output += "\t" + str(el)
            output += "\n"

        return output

    def __mul__(self, other) -> "Matrix":
        if isinstance(other, Matrix):
            a_rows, a_cols = self.dimensions()
            b_rows, b_cols = other.dimensions()
            if a_rows != b_rows and a_cols != b_cols:
                raise ValueError(
                    "Can not scalar multiply matrices with different dimensions."
                )

            result = [
                [
                    self.__matrix[i][j] * other.__matrix[i][j]
                    for j in range(self.__dimensions[1])
                ]
                for i in range(self.__dimensions[0])
            ]

            return Matrix(source=result)

        elif isinstance(other, (int, float)):
            result = [[el * other for el in row] for row in self.__matrix]
            return Matrix(source=result)

        else:
            raise (
                TypeError(
                    f'Can\'t neither multiply Matrix element-wise, nor by scalar with multiplier of type "{
                        type(other)
                    }".'
                )
            )

    def __matmul__(self, other: "Matrix") -> "Matrix":
        a_rows, a_cols = self.dimensions()
        b_rows, b_cols = other.dimensions()

        if a_cols != b_rows:
            raise ValueError("Can't multiply matrices with incompitable dimensions.")

        result = [
            [
                sum(self.__matrix[i][k] * other.__matrix[k][j] for k in range(a_cols))
                for j in range(b_cols)
            ]
            for i in range(a_rows)
        ]

        return Matrix(source=result)

    def __gauss_det(self):
        mat = [row[:] for row in self.__matrix]
        n = self.__dimensions[0]
        swap_count = 0

        for i in range(n):
            max_row = i
            for k in range(i + 1, n):
                if abs(mat[k][i]) > abs(mat[max_row][i]):
                    max_row = k

            if i != max_row:
                mat[i], mat[max_row] = mat[max_row], mat[i]
                swap_count += 1  # Each row swap flips the determinant's sign

            if mat[i][i] == 0:
                return 0.0

            # Perform row reduction
            for k in range(i + 1, n):
                factor = mat[k][i] / mat[i][i]
                for j in range(i, n):
                    mat[k][j] -= factor * mat[i][j]

        # Compute determinant as product of diagonal elements
        det = 1.0
        for i in range(n):
            det *= mat[i][i]

        # Adjust sign based on number of row swaps
        return det if swap_count % 2 == 0 else -det

    def __minor_det(self):
        pass

    def determinant(self, computing_type="gauss"):
        if self.__dimensions[0] != self.__dimensions[1]:
            raise ValueError("Can't compute determinant ")

        match computing_type:
            case "gauss":
                return self.__gauss_det()
            case "minor":
                return self.__minor_det()
            case _:
                raise TypeError(f"Unknown computing type '{computing_type}'")

    def dimensions(self) -> tuple[int, int]:
        return self.__dimensions


def eval_lin_sys(coefficients: Matrix, results: Matrix):
    n, _ = coefficients.dimensions()

    aug_matrix = [[0 for _ in range(n + 1)] for _ in range(n)]
    for i in range(n):
        for j in range(n + 1):
            aug_matrix[i][j] = coefficients[i, j] if j < n else results[0, i]

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(aug_matrix[k][i]) > abs(aug_matrix[max_row][i]):
                max_row = k

        aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]

        if aug_matrix[i][i] == 0:
            return None 

        for k in range(i + 1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n + 1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    solution = [0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = aug_matrix[i][-1]
        for j in range(i + 1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] /= aug_matrix[i][i]

    return solution
