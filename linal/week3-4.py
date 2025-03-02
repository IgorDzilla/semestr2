import lin
import numpy as np


def author_methods(
    vector_1: list, vector_2: list, vector_3: list, computing_type="gramm"
):
    if computing_type == "gramm":
        v1 = lin.Matrix(source=[vector_1])
        v2 = lin.Matrix(source=[vector_2])
        v3 = lin.Matrix(source=[vector_3])
        vectors = [v1, v2, v3]
        gramm = lin.Matrix((len(vectors), len(vectors)))

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                gramm[i, j] = sum(vectors[i] * vectors[j])

        print("GRAMM MATRIX")
        print(gramm)
        volume = np.sqrt(float(gramm.determinant()))
        print("VOLUME WITH GRAMM MATRIX\t", volume)

        return volume

    elif computing_type == "mixed-product":
        matrix = lin.Matrix(source=[vector_1, vector_2, vector_3])
        print("MIXED PRODUCT MUTIPLICATION MATRIX")
        print(matrix)
        volume = matrix.determinant()
        print("VOLUME WITH MIXED PR. MTR\t", volume)
        return volume


def numpy_methods(
    vector_1: list, vector_2: list, vector_3: list, computing_type="gramm"
):
    if computing_type == "gramm":
        vectors = [np.array(vector_1), np.array(vector_2), np.array(vector_3)]
        gramm = np.zeros((len(vectors), len(vectors)))

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                gramm[i, j] = sum(vectors[i] * vectors[j])

        print("GRAMM MATRIX")
        print(gramm)
        volume = np.sqrt(np.linalg.det(gramm))
        print("VOLUME WITH GRAMM MATRIX\t", volume)
        return volume

    elif computing_type == "mixed-product":
        matrix = np.array([vector_1, vector_2, vector_3])
        print("MIXED PRODUCT MUTIPLICATION MATRIX")
        print(matrix)
        volume = np.linalg.det(matrix)
        print("VOLUME WITH MIXED PR. MTR\t", volume)
        return volume


def compute_volume(vector_1, vector_2, vector_3, computing_type="gramm", nplib=False):
    if not nplib:
        return author_methods(vector_1, vector_2, vector_3, computing_type)
    else:
        return numpy_methods(vector_1, vector_2, vector_3, computing_type)


if __name__ == "__main__":
    print("-----------AUTHOR METHODS-----------")
    v1 = author_methods([1, 0, 0], [1, -1, 1], [1, -1, -1])
    print()
    v2 = author_methods(
        [1, 0, 0], [1, -1, 1], [1, -1, -1], computing_type="mixed-product"
    )
    print("\n\n-----------NUMPY METHODS-----------")
    v3 = numpy_methods([1, 0, 0], [1, -1, 1], [1, -1, -1])
    v4 = author_methods(
        [1, 0, 0], [1, -1, 1], [1, -1, -1], computing_type="mixed-product"
    )

    print("\n\n-----------RESULTS-----------")
    print("GRAMM TYPE DIFFERENCE\t", np.abs(v1 - v3))
    print("MIXED MULT PR DIFF.\t", np.abs(v2 - v4))

    print("\n----------------------------\n")

    print("########### SOLVING SYSTEM OF LE ###########")
    print("COEFFS")
    A = lin.Matrix(source=[[2, 3, -1], [4, 7, 2], [6, 18, 6]])
    print(A)
    B = lin.Matrix(source=[[5, 10, 18]])
    print("RESULTS")
    print(B)

    print("-----------AUTHOR METHODS-----------")
    res_A = lin.eval_lin_sys(A, B)
    print(res_A)

    print("-----------NUMPY-----------")
    A = np.array([[2, 3, -1], [4, 7, 2], [6, 18, 6]])
    B = np.array([5, 10, 18])

    res = np.linalg.solve(A, B)
    print(res)
