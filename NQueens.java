import java.util.*;

public class NQueens {
    int n;
    int[][] board;

    NQueens(int n) {
        this.n = n;
        board = new int[n][n];
    }

    boolean isSafe(int row, int col) {
        // Check row on left
        for (int i = 0; i < col; i++)
            if (board[row][i] == 1)
                return false;

        // Upper-left diagonal
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1)
                return false;

        // Lower-left diagonal
        for (int i = row, j = col; j >= 0 && i < n; i++, j--)
            if (board[i][j] == 1)
                return false;

        return true;
    }

    boolean solve(int col) {
        if (col == n)    // all queens placed
            return true;

        for (int row = 0; row < n; row++) {

            if (board[row][col] == 1) {   // fixed queen already here
                if (isSafe(row, col))
                    return solve(col + 1);
                else
                    return false;
            }

            if (isSafe(row, col)) {
                board[row][col] = 1;   // place queen

                if (solve(col + 1))
                    return true;

                board[row][col] = 0;   // backtrack
            }
        }
        return false;
    }

    void printBoard() {
        for (int[] row : board) {
            for (int cell : row)
                System.out.print(cell == 1 ? "Q " : "X ");
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter n: ");
        int n = sc.nextInt();

        NQueens obj = new NQueens(n);

        System.out.print("Enter fixed Queen row (1-" + n + "): ");
        int r = sc.nextInt();

        System.out.print("Enter fixed Queen column (1-" + n + "): ");
        int c = sc.nextInt();

        // Place fixed queen
        obj.board[r - 1][c - 1] = 1;

        // Start solving from column 0
        if (obj.solve(0)) {
            System.out.println("\nSolution:");
            obj.printBoard();
        } else {
            System.out.println("No solution possible with this fixed Queen.");
        }
    }
}
