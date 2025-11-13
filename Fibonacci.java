import java.util.*;

class Fibonacci {
    static int fibIter(int n) {
        if (n == 1) return 0;
        if (n == 2) return 1;
        int a = 0, b = 1, c = 0;
        for (int i = 3; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }

    static int fibRec(int n) {
        if (n == 1) return 0;
        if (n == 2) return 1;
        return fibRec(n - 1) + fibRec(n - 2);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter n: ");
        int n = sc.nextInt();
        System.out.println("Fibonacci (Iterative): " + fibIter(n));
        System.out.println("Fibonacci (Recursive): " + fibRec(n));
    }
}