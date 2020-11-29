import numpy as np
import numpy.linalg as LA


RED = '\33[31m'
GREEN = '\33[32m'
END = '\033[0m'


class LinearLeastSquares:
    """Provides a solution x with different methods."""
    def __init__(self, n):
        self.x = np.random.randn(n, 1)

    def __call__(self, A):
        return A @ self.x

    def grad(self, A, b):
        return A.T @ self(A) - A.T @ b

    def gradient_descent(self, A, b, step_size, tolerance):
        while LA.norm(grad := self.grad(A, b)) > tolerance:
            self.x -= step_size * grad

    def moore_penrose_pseudoinverse(self, A, b):
        self.x = LA.pinv(A) @ b

    def system_of_linear_equations(self, A, b):
        ATA = A.T @ A
        if LA.matrix_rank(ATA) != ATA.shape[0]:
            print(RED + 'System of linear equations not uniquely solvable.' + END)
            # This means that the loss function
            # either has infinitely many critical points or
            # it has none.
            self.x = LA.solve(A.T @ A, A.T @ b)
            # This gives one solution to the system if it has any.
        else:
            self.x = LA.inv(ATA) @ A.T @ b
            # We can use this formula as the matrix is invertible.
            # (We could of course use LA.solve, too.)


class ConstraintedLinearLeastSquares:
    """Provides a solution x with the constraint x.T @ x == 1."""
    def gradient_ascent(self, A, b, step_size, tolerance):
        lambda_ = 0
        self.x = LA.solve(A.T @ A, A.T @ b)

        while grad_lambda := self.x.T @ self.x - 1 > tolerance:
            lambda_ += step_size * grad_lambda
            I = np.eye(A.shape[1])
            self.x = LA.solve(A.T @ A + 2 * lambda_ * I, A.T @ b)
        

def main():
    lls_1 = LinearLeastSquares(3)
    A_1 = np.array([[4, 5, 1], [1, 2, 5], [0, -3, 2], [1, 1, 1]])
    b_1 = np.array([1, 5, 4, 3]).reshape((4, 1))

    lls_2 = LinearLeastSquares(4)
    A_2 = A_1.T
    b_2 = b_1[:3]

    for lls, A, b in zip([lls_1, lls_2], [A_1, A_2], [b_1, b_2]):
        print(GREEN + '-' * 50 + END)
        print(f'{A=!s}')
        print(f'{b=!s}')
        lls.gradient_descent(A, b, 0.01, 1e-13)
        print(GREEN + 'Gradient descent:' + END)
        print(lls.x)
        print(GREEN + 'Gives the solution:' + END)
        print(A @ lls.x)

        lls.moore_penrose_pseudoinverse(A, b)
        print(GREEN + 'Moore-Penrose pseudoinverse:' + END)
        print(lls.x)
        print(GREEN + 'Gives the solution:' + END)
        print(A @ lls.x)

        try:
            lls.system_of_linear_equations(A, b)
            print(GREEN + 'System of linear equations:' + END)
            print(lls.x)
            print(GREEN + 'Gives the solution:' + END)
            print(A @ lls.x)
        except LA.LinAlgError:
            print(RED + 'System of linear equations not solveable.' + END)

    print(GREEN + '-' * 50 + END)
    A, b = A_1, b_1
    print(f'{A=!s}')
    print(f'{b=!s}')
    clls = ConstraintedLinearLeastSquares()
    clls.gradient_ascent(A, b, 0.01, 0)
    print(GREEN + 'Gradient ascent with constrained optimization:' + END)
    print(clls.x)
    print(GREEN + '... gives the solution:' + END)
    print(A @ clls.x)


if __name__ == "__main__":
    main()
