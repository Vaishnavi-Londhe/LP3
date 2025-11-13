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