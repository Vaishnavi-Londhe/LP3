# LP3
# 1 NonRecursiveFibonacci

import java.util.Scanner;
public class NonRecursiveFibonacci {

    // Non-recursive (iterative) Fibonacci function
    public static void fibonacci(int n) {
        int n1 = 0, n2 = 1, n3;

        // Print first two terms
        if (n > 0)
            System.out.print(n1 + " ");
        if (n > 1)
            System.out.print(n2 + " ");

        // Loop to calculate next Fibonacci numbers
        for (int i = 2; i < n; i++) {
            n3 = n1 + n2;
            System.out.print(n3 + " ");
            n1 = n2;
            n2 = n3;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the number of terms: ");
        int n = sc.nextInt();

        System.out.println("\nFibonacci Series (Non-Recursive): ");
        long start = System.nanoTime(); // Start time measurement

        fibonacci(n); // Function call

        long end = System.nanoTime();   // End time measurement
        double timeTaken = (end - start) / 1000.0; // microseconds

        // ---------- Time & Space Complexity Analysis ----------
        System.out.println("\n\n=== Time and Space Complexity Analysis ===");
        System.out.printf("Time Taken: %.2f microseconds%n", timeTaken);
        System.out.println("Time Complexity: O(n)");
        System.out.println("Space Complexity: O(1)");

        sc.close();
    }
}

/*
---------------------------------------------
    ANALYSIS OF TIME AND SPACE COMPLEXITY
---------------------------------------------

1️⃣ TIME COMPLEXITY:
   - The loop runs from 2 to n.
   - Each iteration does a constant amount of work (addition and assignment).
   - Hence:
        Time Complexity = O(n)

2️⃣ SPACE COMPLEXITY:
   - Uses only a few integer variables (n1, n2, n3).
   - No recursion or extra data structures.
   - Hence:
        Space Complexity = O(1)

✅ FINAL SUMMARY:
   Approach          : Non-Recursive (Iterative)
   Time Complexity   : O(n)
   Space Complexity  : O(1)
---------------------------------------------
*/

# 2
# 3 HuffmanEncoding

import java.util.*;
class Node {
    char ch;
    int freq;
    Node left, right;

    Node(char c, int f) {
        ch = c;
        freq = f;
    }

    Node(Node l, Node r) {
        left = l;
        right = r;
        freq = l.freq + r.freq;
    }
}

class HuffmanEncoding {
    static Map<Character, String> codes = new HashMap<>();

    static void buildCode(Node root, String s) {
        if (root == null) return;

        // Leaf node (character found)
        if (root.left == null && root.right == null) {
            codes.put(root.ch, s);
            return;
        }

        buildCode(root.left, s + "0");
        buildCode(root.right, s + "1");
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter string: ");
        String s = sc.nextLine();

        // Step 1: Frequency map
        Map<Character, Integer> freqMap = new HashMap<>();
        for (char c : s.toCharArray())
            freqMap.put(c, freqMap.getOrDefault(c, 0) + 1);

        // Step 2: Min-Heap
        PriorityQueue<Node> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.freq));

        for (Map.Entry<Character, Integer> e : freqMap.entrySet())
            pq.add(new Node(e.getKey(), e.getValue()));

        // Step 3: Build Huffman Tree
        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();
            pq.add(new Node(left, right));
        }

        Node root = pq.peek();

        // Step 4: Generate Huffman Codes
        buildCode(root, "");

        // Step 5: Display codes
        System.out.println("\nCharacter | Huffman Code");
        for (Map.Entry<Character, String> e : codes.entrySet())
            System.out.println(e.getKey() + " -> " + e.getValue());
    }
}
# 4 FractionalKnapsack

import java.util.*;
class Item {
    int value, weight;
    Item(int v, int w) {
        value = v;
        weight = w;
    }
}

public class FractionalKnapsack {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        Item[] items = new Item[n];

        System.out.println("Enter value and weight of each item:");
        for (int i = 0; i < n; i++) {
            System.out.print("Item " + (i + 1) + " value: ");
            int v = sc.nextInt();
            System.out.print("Item " + (i + 1) + " weight: ");
            int w = sc.nextInt();
            items[i] = new Item(v, w);
        }

        System.out.print("Enter capacity of knapsack: ");
        int capacity = sc.nextInt();

        // Step 1: Sort items by value/weight ratio (descending order)
        Arrays.sort(items, (a, b) -> Double.compare((double)b.value / b.weight, (double)a.value / a.weight));

        double totalValue = 0.0;

        // Step 2: Pick items greedily
        for (int i = 0; i < n; i++) {
            if (capacity >= items[i].weight) {
                // Take the whole item
                totalValue += items[i].value;
                capacity -= items[i].weight;
            } else {
                // Take fraction of the item
                double fraction = (double) capacity / items[i].weight;
                totalValue += items[i].value * fraction;
                break; // Knapsack is full
            }
        }

        // Step 3: Print result
        System.out.println("\nMaximum value in Knapsack = " + totalValue);
    }
}

# 5 Knapsack_0_1

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

# 6 NQueens
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

# ML
# 1 uber
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
#We do not want to see warnings
warnings.filterwarnings("ignore") 

#import data
data = pd.read_csv("uber.csv")
#Create a data copy
df = data.copy()
#Print data
df.head()
#Get Info
df.info()
#pickup_datetime is not in required data format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.info()

#Statistics of data
df.describe()
#Number of missing values
df.isnull().sum()

#Correlation
df.select_dtypes(include=[np.number]).corr()

print(df.columns)

#Drop the rows with missing values
df.dropna(inplace=True)

plt.boxplot(df['fare_amount'])
#Remove Outliers
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)

df = df[(df["fare_amount"] < q_hi) & (df["fare_amount"] > q_low)]
#Check the missing values now
df.isnull().sum()

#Time to apply learning models
from sklearn.model_selection import train_test_split
#Take x as predictor variable
x = df.drop("fare_amount", axis = 1)
#And y as target variable
y = df['fare_amount']
#Necessary to apply model
x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

#Prediction
predict = lrmodel.predict(x_test)
#evaluation

from sklearn.metrics import mean_squared_error, r2_score

lr_rmse = np.sqrt(mean_squared_error(y_test, predict))
lr_r2 = r2_score(y_test, predict)

print("Linear Regression → RMSE:", lr_rmse, "R²:", lr_r2)

#Let's Apply Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(n_estimators = 100, random_state = 101)
#Fit the Forest
rfrmodel.fit(x_train, y_train)
rfrmodel_pred = rfrmodel.predict(x_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfrmodel_pred))
rfr_r2 = r2_score(y_test, rfrmodel_pred)

print("Random Forest → RMSE:", rfr_rmse, "R²:", rfr_r2)
