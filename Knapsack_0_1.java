import java.util.*;

class Knapsack_0_1 {
    static int knapsack(int[] val, int[] wt, int cap) {
        int n = val.length;
        int[][] dp = new int[n + 1][cap + 1];
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= cap; w++) {
                if (wt[i - 1] <= w)
                    dp[i][w] = Math.max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
                else
                    dp[i][w] = dp[i - 1][w];
            }
        }
        return dp[n][cap];
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter number of items: ");
        int n = sc.nextInt();
        int[] val = new int[n];
        int[] wt = new int[n];

        System.out.print("Enter values: ");
        for (int i = 0; i < n; i++) val[i] = sc.nextInt();

        System.out.print("Enter weights: ");
        for (int i = 0; i < n; i++) wt[i] = sc.nextInt();

        System.out.print("Enter capacity: ");
        int cap = sc.nextInt();

        System.out.println("Maximum Value: " + knapsack(val, wt, cap));
    }
}
