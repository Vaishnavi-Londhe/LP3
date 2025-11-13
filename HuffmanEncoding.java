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
