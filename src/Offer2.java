import java.util.*;

/**
 * User: gaohan
 * Date: 2021/8/11
 * Time: 17:38
 * 专项训练
 */
public class Offer2 {

  // 两数之和
  /**
   * 剑指 Offer II 006. 排序数组中两数之和
   */
  public int[] twoSum(int[] numbers, int target) {
    if(numbers==null || numbers.length == 0){
      return null;
    }
    int[] res = new int[2];
    int len = numbers.length;
    int left = 0;
    int right = len - 1;
    while(left < right){
      if(numbers[left] + numbers[right] == target){
        res[0] = left;
        res[1] = right;
        return res;
      } else if(numbers[left] + numbers[right] < target){
        left++;
      } else {
        right--;
      }
    }
    return null;
  }

  /**
   * 剑指 Offer II 007. 数组中和为 0 的三个数
   */
  public List<List<Integer>> threeSum(int[] nums) {
    if(nums==null || nums.length < 3){
      return new ArrayList<>();
    }
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    int sum = 0 ;
    for(int i = 0; i<nums.length; i++){
      sum = -nums[i];
      if(sum < 0) break;
      if(i>0 && nums[i] == nums[i-1]) continue;
      int left = i+1;
      int right = nums.length - 1;
      while(left < right){
        if(nums[left]+nums[right]==sum){
          List<Integer> list = new ArrayList<>();
          list.add(nums[i]);
          list.add(nums[left]);
          list.add(nums[right]);
          res.add(list);
          while(left<right && nums[left+1] == nums[left]){
            left++;
          }
          while(left > right && nums[right-1] == nums[right]){
            right--;
          }
          left++;
          right--;
        } else if(nums[left] + nums[right] > sum){
          right--;
        } else{
          left++;
        }
      }
    }
    return res;
  }

  //---------------子序列/子数组
  /**
   *剑指 Offer II 008. 和大于等于 target 的最短子数组
   * 最短连续子数组 和大于等于target
   */
  public int minSubArrayLen(int target, int[] nums) {
    if(nums == null || nums.length == 0){
      return 0;
    }
    int l = 0;
    int r = 0;
    int ans = nums.length + 1;
    int sum = 0;
    while(r < nums.length){
      sum += nums[r];
      while(sum >= target){
        ans = Math.min(ans, r-l+1);
        sum -= nums[l++];
      }
      r++;

    }
    return ans>nums.length?0:ans;
  }

  /**
   * 剑指 Offer II 009. 乘积小于 K 的子数组
   * 乘积小于 K 的子数组的数组个数
   */
  public int numSubarrayProductLessThanK(int[] nums, int k) {
    if(nums == null || nums.length == 0){
      return 0;
    }
    int res = 0;
    int mul = 1;
    for(int i = 0; i<nums.length; i++){
      mul = nums[i];
      if(mul < k){
        res++;
      } else {
        continue;
      }
      for(int j = i+1; j<nums.length; j++){
        mul *= nums[j];
        if(mul<k){
          res ++;
        } else {
          break;
        }
      }

    }
    return res;
  }

  /**
   * 剑指 Offer II 010. 和为 k 的子数组
   * 和为K的连续子数组的个数
   * 前缀和：记录sum,和sum出现的次数
   */
  //前缀和的方法，前n个数和为sum, 前m个数和为k， 则m-n个数的和sum-k,
  public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>(); // 表示的是
    int sum = 0;
    int res = 0;
    map.put(0,1);
    for(int i = 0; i< nums.length; i++){
      sum += nums[i];
      if(map.containsKey(sum-k)){
        res += map.get(sum-k);
      }
      map.put(sum, map.getOrDefault(sum,0)+1);
    }
    return res;


  }
  public int subarraySum1(int[] nums, int k) {
    if(nums== null || nums.length == 0){
      return 0;
    }
    int sum = 0;
    int res = 0;
    for(int i = 0; i<nums.length; i++){
      sum = 0;
      for(int j = i; j<nums.length; j++){
        sum += nums[j];
        if(sum == k){
          res++;
        }
      }

    }
    return res;
  }


    /**
     * 剑指 Offer II 011. 0 和 1 个数相同的子数组
     * 暴力的方法， 利用两次遍历，第一次遍历确定初始位置，第二次遍历确定末尾位置， 可能超时
     * 第二种是使用前缀和的方法，用map记录sum和首次出现这个sum的下标，在此出现这个sum时，说明这两个sum之间的和为0
     */
  public int findMaxLength(int[] arr) {
    int sum = 0;
    int max = 0;
    for(int i = 0;i<arr.length; i++){
      sum = arr[i] == 0?-1:1;
      for(int j = i+1; j<arr.length; j++) {
        sum += arr[j] ==0?-1:1;
        if(Math.abs(sum)>arr.length-j || max > arr.length - i ){
          break;
        }
        if(sum == 0){
          max = Math.max(max, j-i+1);
        }
      }
    }
    return max;
  }
  public int findMaxLength1(int[] arr){
    Map<Integer, Integer> map = new HashMap<>(); //表示以key:sum value:index。map存储sum,和首次出现时的下标
    int sum = 0;
    int res = 0;
    map.put(0, -1); //base
    for (int i = 0; i < arr.length; i++) {
      sum += arr[i] == 0?-1:1;
      if(map.containsKey(sum)){ //如果第二次遇到这个sum,说明这两个sum之间的数和为0，也就是0，1的数量相同
        res = Math.max(res, i - map.get(sum));
      } else {
        map.put(sum, i);
      }

    }
    return  res;
  }
  /**
   *剑指 Offer II 012. 左右两边子数组的和相等
   *
   */
  public int pivotIndex(int[] nums) {
    if(nums == null || nums.length == 0){
      return  -1;
    }
    int[] A = new int[nums.length];
    int[] B = new int[nums.length];
    A[0] = 0;
    B[nums.length - 1] = 0;
    for(int i = 1; i<nums.length; i++){
      A[i] = A[i-1]+nums[i-1];
    }
    for(int i=nums.length-2; i>=0;i--){
      B[i] = B[i+1] + nums[i+1];
    }
    for(int i = 0; i<nums.length; i++){
      if(A[i] == B[i]){
        return i;
      }
    }
    return -1;

  }
  //还可以前缀和
  public int pivotIndex1(int[] nums) {
    if(nums == null || nums.length == 0){
      return  -1;
    }
    int[] sum = new int[nums.length+1];//sum[i]表示所有左边的和
    sum[0] = 0;
    for(int i = 1; i<=nums.length; i++){
      sum[i] = sum[i-1] + nums[i-1];
    }
    int max = sum[nums.length];//所有的数的和
    for(int i = 0; i<nums.length; i++){
      if(max-nums[i] == 2*sum[i]){
        return i;
      }

    }
    return -1;

  }

  /**
   * 剑指 Offer II 013. 二维子矩阵的和
   * 二维前缀和
   */

  /**
   * 剑指 Offer II 014. 字符串中的变位词
   */
  public boolean checkInclusion(String s1, String s2) {
    int[] A = new int[26];
    int[] B = new int[26];
    if(s2.length() < s1.length()){
      return false;
    }
    for (int i = 0; i < s1.length(); i++) {
      A[s1.charAt(i)-'a']++;
      B[s2.charAt(i)-'a']++;
    }
    if(Arrays.equals(A,B)){
      return true;
    }

    int left = 0;
    int right = s1.length();
    while(right < s2.length()){
      B[s2.charAt(left++) -'a']--;
      B[s2.charAt(right++)-'a']++;
      if(Arrays.equals(A, B)){
        return  true;
      }
    }

    return false;

  }

  /**
   * 剑指 Offer II 015. 字符串中的所有变位词
   */
  public List<Integer> findAnagrams(String s, String p) {
    int l1 = s.length();
    int l2 = p.length();
    if(l2>l1){
      return null;
    }
    int[] P = new int[26];
  // int[] S = new int[26];
    List<Integer> res = new ArrayList<>();
    for (int i = 0; i < p.length(); i++) {
      P[p.charAt(i)-'a']++;
     // S[s.charAt(i)-'a']++;
    }
    for(int i = 0; i<l1;i++){
      int[] S = new int[26];
      for(int j = 0; j<l2&&i+j<l1; j++){
        S[s.charAt(i+j) -'a']++;
      }
      if(Arrays.equals(S,P)){
        res.add(i);
      }
    }
//    int left = 0;
//    int right = l1;
//    while(right < l2){
//      S[s.charAt(left++)-'a']--;
//      S[s.charAt(right++)-'a']++;
//      if(Arrays.equals(S,P)){
//        res.add(left);
//      }
//    }
    return  res;

  }

  /**
   * 剑指 Offer II 016. 不含重复字符的最长子字符串
   * 滑动窗口，更新左指针
   * TODO
   */
  public static int lengthOfLongestSubstring(String s) {
    int len = s.length();
    int l = -1;
    int res = 0;
    HashMap<Character, Integer> hashMap = new HashMap<>();
    for (int i = 0; i < len; i++) {
      if(hashMap.containsKey(s.charAt(i))){
        l = Math.max(hashMap.get(s.charAt(i)), l);
      }
      hashMap.put(s.charAt(i),i);
      res = Math.max(res, i-l);
    }
    return res;
  }
  /**
   * 牛客：最长无重复子数组
   * 用一个哈希表记录每次每个元素最新出现的位置，如果出现了相同的，就更新左边界，记录最长的窗口
   */
  public int maxLength (int[] arr) {
    int len = arr.length;
    int l = -1;
    int res = 0;
    HashMap<Integer, Integer> hashMap = new HashMap<>();
    for (int i = 0; i < len; i++) {
      if(hashMap.containsKey(arr[i])){
        l = Math.max(hashMap.get(arr[i]), l);
      }
      hashMap.put(arr[i], i);
      res = Math.max(res, i-l);
    }
    return res;

  }


  /**
   *剑指 Offer II 018. 有效的回文
   */
  public boolean isPalindrome(String s) {
    int left = 0;
    int right = s.length() - 1;
    while(left < right){
      while(left< right &&!isValid(s.charAt(left))) left++;
      while(left < right && !isValid(s.charAt(right))) right--;
      if(Character.toUpperCase(s.charAt(left)) != Character.toUpperCase(s.charAt(right))){
        return false;
      }
      left++;
      right--;
    }
    return true;

  }
  private boolean isValid(char c){
    if((c >='a' && c<='z') || (c >='A' && c<='Z') ){
      return true;
    } else {
      return false;
    }
  }

  /**
   * 剑指 Offer II 019. 最多删除一个字符得到回文
   *
   */
  public boolean validPalindrome(String s) {
    int left = 0;
    int right = s.length() - 1;
    while(left < right){
      if(s.charAt(left) != s.charAt(right)){
        if(isPalindrme(s, left+1, right) || isPalindrme(s, left, right-1)) {
          return true;
        } else {
          return  false;
        }

      }
      left++;
      right--;
    }
    return true;

  }
  private boolean isPalindrme(String s, int i , int j){
    while(i<j){
      if(s.charAt(i) != s.charAt(j)){
        return false;
      }
      i++;
      j--;
    }
    return true;
  }

  /**
   * 剑指 Offer II 020. 回文子字符串的个数
   * 暴力法：
   * 优化：中心扩散法 动态规划
   * dp[i][j] 表示i-j之间是否时候回文，dp[i][j]依赖dp[i+1][j-1] todo
   */
  public int countSubstrings(String s) {
    int res = 0;
    for (int i = 0; i < s.length(); i++) {
      for (int j = i; j < s.length(); j++) {
        if(isPalindrme(s, i, j)){
          res++;
        }
      }
    }
    return  res;
  }

  /**
   * 5. 最长回文字符串
   */
  public String longestPalindrome(String s) {
    int len = s.length();
    boolean[][] dp = new boolean[len][len];
    for (int i = 0; i < len; i++) {
      dp[i][i] = true;
    }
    int max = 1;
    int start = 0;
    for (int l = 2; l <= len; l++) { // 枚举子串的长度
      for(int i = 0; i<len; i++){ //枚举子串开始的位置
        int j = i+l-1;
        if(j>=len){
          break;
        }
        if(s.charAt(i) != s.charAt(j)){
          dp[i][j] = false;
        } else {
          if(j-i+1 < 3){
            dp[i][j] = true;
          } else {
            dp[i][j] = dp[i+1][j-1];
          }
        }
        if(dp[i][j] && j-i+1 > max){
          max = j-i+1;
          start = i;
        }

      }

    }
    return  s.substring(start, start+max+1);

  }


    //------------链表---------------------
  public class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

  /**
   * 剑指 Offer II 021. 删除链表的倒数第 n 个结点
   */
  public ListNode removeNthFromEnd(ListNode head, int n) {
    if(head == null ){
      return head;
    }
    ListNode slow = head;
    ListNode fast = head;
    int idx = 0;
    while(fast!=null && idx<n){
      fast = fast.next;
      idx++;
    }
    if(fast == null){ //说明删除的是头节点
      return head.next;
    }
    while(fast.next != null){
      slow = slow.next;
      fast = fast.next;
    }
    ListNode next = slow.next.next;
    slow.next = next;
    return  head;
  }

  /**
   * 剑指 Offer II 022. 链表中环的入口节点
   */
  public ListNode detectCycle(ListNode head) {
    if(head == null || head.next == null){
      return head;
    }
    ListNode slow = head;
    ListNode fast = head;
    while(true){
      if(fast == null || fast.next == null){
        return  null;
      }
      slow = slow.next;
      fast = fast.next.next;
      if(fast == slow) break;
    }
    ListNode pre = head;
    while(pre != fast){
       pre = pre.next;
       fast = fast.next;
    }
    return  pre;
  }

    /**
     * 剑指 Offer II 023. 两个链表的第一个重合节点
     * 公共节点
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
      if(headA == null || headB == null){
        return  null;
      }
      ListNode cur1 = headA;
      ListNode cur2 = headB;
      while(cur1 != cur2){
        cur1 = cur1 != null?cur1.next:headB;
        cur2 = cur2 != null?cur2.next:headA;
      }
      return  cur1;
    }

    /**
     * 剑指 Offer II 024. 反转链表
     */
    public ListNode reverseList(ListNode head) {
      if(head == null || head.next == null){
        return head;
      }
      ListNode pre = null;
      ListNode cur = head;
      while(cur != null){
        ListNode tmp = cur.next;
        cur.next = pre;
        pre = cur;
        cur = tmp;
      }
      return pre;
    }

    //递归的思想：把head之后的进行逆序排列，然后更改head 和head next 之间的转向
    public ListNode reverse(ListNode head){
      if(head == null || head.next == null){
        return head;
      }
      ListNode cur = reverse(head.next);
      head.next.next = head;
      head.next = null;
      return  cur;
    }

    /**
     * 剑指 Offer II 025. 链表中的两数相加
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
      ListNode node1 = reverseList(l1);
      ListNode node2 = reverseList(l2);
      int carry = 0;
      ListNode l3 = new ListNode(-1);
      ListNode pre = l3;
      while(node1 != null || node2 != null || carry > 0){
        int val1 = node1==null?0:node1.val;
        int val2 = node2==null?0:node2.val;
        int sum = val1+val2+carry;
        ListNode node = new ListNode(sum%10);
        carry = sum/10;
        pre.next = node;

        node1 = node1==null?null:node1.next;
        node2 = node2==null?null:node2.next;
        pre = node;
      }
      return reverseList(l3.next);
    }

    /**
     * 剑指 Offer II 026. 重排链表
     * 利用栈记录从尾来的链表
     */
    public void reorderList(ListNode head) {
      if(head == null || head.next == null){
        return;
      }
      ListNode cur = head;
      Stack<ListNode> stack = new Stack<>();
      int cnt = 0;
      while(cur != null){
        cnt++;
        stack.push(cur);
        cur = cur.next;
      }
      cur = head;
      for(int i = 0; i<cnt/2; i++){
        ListNode top = stack.pop();
        ListNode tmp = cur.next;
        cur.next = top;
        top.next = tmp;
        cur = tmp;
      }
      cur.next = null;
    }

      /**
       * 剑指 Offer II 027. 回文链表
       */
      public boolean isPalindrome(ListNode head) {
        if(head == null || head.next == null){
          return true;
        }
        ListNode cur = head;
        Stack<Integer> stack = new Stack<>();
        int cnt = 0;
        while(cur != null){
          cnt++;
          stack.push(cur.val);
          cur = cur.next;
        }
        cur = head;
        int top = stack.pop();
        for(int i = 0; i<cnt/2; i++){
          if(cur.val != top){
            return false;
          } else{
            cur = cur.next;
            top = stack.pop();
          }
        }
        return true;

      }

      /**
       * 剑指 Offer II 028. 展平多级双向链表
       */
      class Node {
        public int val;
        public Node prev;
        public Node next;
        public Node child;
        Node(int v){val = v;};
      };
      Node cur = new Node(-1);
      Node res = cur;
      public Node flatten(Node head) {
        if(head == null){
          return head;
        }
        func(head);
        res.next.prev = null;
        return res.next;
      }
      private void func(Node head){
        if(head == null){
          return;
        }
        Node child = head.child;
        Node next = head.next;
        cur.next = head;
        cur.next.prev = cur;
        cur = cur.next;
        cur.child = null;
        func(child);
        func(next);

      }

      /**
       * 剑指 Offer II 029. 排序的循环链表
       */
      public Node insert(Node head, int insertVal) {
        if(head == null){
          Node newHead = new Node(insertVal);
          return newHead;
        }
        //先找到链表真正的头节点
        Node cur = head;
        Node next = cur.next;
        while(cur.val <= next.val){
          cur = cur.next;
          next = next.next;
          if(cur == next) break;
        }
        Node trueHead = next;
        while(next.val < insertVal){
          cur = next;
          next = next.next;
          if(next == trueHead) break;
        }
        cur.next = new Node(insertVal);
        cur = cur.next;
        cur.next = next;
        return head;

      }
  /**
   * 剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器
   */
  class RandomizedSet {
    private HashMap<Integer, Integer> num_idx = new HashMap<>();
    private List<Integer> nums = new ArrayList<>();
    Random rand = new Random();


    /** Initialize your data structure here. */
    public RandomizedSet() {

    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
      if(num_idx.containsKey(val)){
        return false;
      }
      int size = nums.size();
      num_idx.put(val, size);
      nums.add(val);
      return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
      if(!num_idx.containsKey(val)){
        return false;
      }
      int idx = num_idx.get(val);
      int last_num = nums.get(nums.size() - 1);
      num_idx.put(last_num, idx); //删除操作时先找到list中的最后一个元素，把最后一个元素放到要移除的位置，然后删除最后一个元素
      nums.set(idx, last_num);
      num_idx.remove(val);
      nums.remove(nums.size() - 1);
      return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
      int idx = rand.nextInt(nums.size());
      return nums.get(idx);
    }
  }

  /**
   * 剑指 Offer II 031. 最近最少使用缓存 LRU cache
   */
  class LRUCache {
    class ListNode{
      private int key;
      private int value;
      ListNode pre;
      ListNode next;
      public ListNode(){};
      public ListNode(int key,int value) {this.key=key;this.value=value;}
    }
    ListNode head;
    ListNode tail;
    int size;
    int capacity;
    Map<Integer,ListNode> map=new HashMap<>();
    public LRUCache(int capacity) {
      this.size=0;
      this.capacity=capacity;
      head=new ListNode();
      tail=new ListNode();
      head.next=tail;
      tail.pre=head;
    }

    public int get(int key) {
      ListNode node=map.get(key);
      if(node==null){
        return -1;
      }else{
        moveTohead(node);
        return node.value;
      }
    }

    public void put(int key, int value) {
      ListNode node=map.get(key);
      if(node==null){
        node=new ListNode(key,value);
        map.put(key,node);
        addTohead(node);
        size++;
        if(size>capacity){
          ListNode res=removeTail();
          map.remove(res.key);
          size--;
        }
      }else{
        node.value=value;
        moveTohead(node);
      }
    }

    public void addTohead(ListNode node){
      node.next=head.next;
      node.next.pre=node;
      head.next=node;
      node.pre=head;
    }

    public void removeNode(ListNode node){
      node.pre.next=node.next;
      node.next.pre=node.pre;
    }

    public void moveTohead(ListNode node){
      removeNode(node);
      addTohead(node);
    }

    public ListNode removeTail(){
      ListNode res=tail.pre;
      removeNode(res);
      return res;
    }
  }


  /**
       * 剑指 Offer II 032. 有效的变位词
       */
      public boolean isAnagram(String s, String t) {
        int l1 = s.length();
        int l2 = t.length();
        if(l1 != l2){
          return false;
        }
        int[] S = new int[26];
        int[] T = new int[26];
        for (int i = 0; i < l1; i++) {
          S[s.charAt(i) - 'a']++;
          T[t.charAt(i) - 'a']++;
        }
        if(Arrays.equals(S,T)){
          return true;
        } else {
          return false;
        }

      }
      /**
       * 剑指 Offer II 033. 变位词组
       */
      public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for(String str : strs){
          char[] chars = str.toCharArray();
          Arrays.sort(chars);
          //key
          String key = new String(chars);
          List<String> value = map.getOrDefault(key, new ArrayList<>());
          value.add(str);
          map.put(key, value);

        }
        return new ArrayList<>(map.values());
      }

      /**
       * 剑指 Offer II 034. 外星语言是否排序
       */
//      public boolean isAlienSorted(String[] words, String order) {
//
//      }
      /**
       * 剑指 Offer II 035. 最小时间差
       */
      public int findMinDifference(List<String> timePoints) {
        List<Integer> list = new ArrayList<>();
        for(String str : timePoints){
          int h = (str.charAt(0)-'0')*10 + (str.charAt(1)-'0');
          int m = (str.charAt(3) -'0')*10 + (str.charAt(4)-'0');
          int t = h*60+m;
          list.add(t);
        }
        Collections.sort(list);
        int res = 1440;
        for (int i = 0; i < list.size()-1; i++) {
          if(list.get(i+1) - list.get(i) < res){
            res = list.get(i+1) - list.get(i);
          }
        }
        return Math.min(res, 1440 - (list.get(list.size()-1) - list.get(0)));

      }


      /**
       * 剑指 Offer II 036. 后缀表达式
       */
      public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
          if(token.equals("+") || token.equals("-") || token.equals("/") || token.equals("*")){
              int i = stack.pop();
              int j = stack.pop();
              if(token.equals("+")){
                stack.push(i+j);
              } else if(token.equals("-")){
                stack.push(j-i);
              } else if(token.equals("*")){
                stack.push(i*j);
              } else {
                stack.push(j/i);
              }
          } else {
            stack.push(Integer.valueOf(token));
          }
        }
        return stack.peek();
      }

      /**
       * 剑指 Offer II 037. 小行星碰撞
       */
      public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int asteroid : asteroids) {
          boolean flag = true; //判断是否将当前的元素压入栈
          while(!stack.isEmpty() && stack.peek()>0 && asteroid<0){
            int t = stack.pop();
            if(asteroid + t > 0){
              stack.push(t);
              flag = false;
              break;
            } else if(asteroid + t == 0) {
              flag = false;
            }
          }
          if(flag) stack.push(asteroid);
        }
        int[] res = new int[stack.size()];
        for(int i = stack.size()-1; i>=0; i--){
          res[i] = stack.pop();
        }
        return res;

      }
      /**
       * 剑指 Offer II 038. 每日温度
       * 暴力
       * 单调栈
       */
      public static int[] dailyTemperatures(int[] temperatures) {
        int len = temperatures.length;
        int[] res = new int[len];
        for (int i = 0; i < len; i++) {
          int cur = temperatures[i];
          for(int j = i+1;j<len;j++){
            if(temperatures[j] > temperatures[i]){
              res[i] = j-i+1;
              break;
            }
          }
        }
        return res;
      }

  //------------二叉树-----------
  public class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int v ){val = v;};
  }
  /**
   * 牛客：二叉树根节点到叶子节点和为指定值的路径
   */
//  List<List<Integer>> res = new ArrayList<>();
//  public List<List<Integer>> pathSum (TreeNode root, int sum) {
//    List<Integer> path = new ArrayList<>();
//    return res;
//
//
//  }
//  private void dfs1(TreeNode root, List<Integer> path, int sum){
//    if(root == null){
//      return;
//    }
//    if(root.left == null && root.right == null && sum - root.val == 0){
//      path.add(root.val);
//      res.add(new ArrayList<>(path));
//      path.remove(path.size() -1);
//      return;
//    }
//    path.add(root.val);
//    dfs1(root.left, path, sum - root.val);
//    dfs1(root.right, path, sum - root.val);
//    path.remove(path.size() - 1);
//  }

  /**
   * 牛客：二叉树的最大路径和
   */
  int max = Integer.MIN_VALUE;
  public int maxPathSum (TreeNode root) {
    getMax(root);
    return max;
  }
  private int getMax(TreeNode root){
    if(root == null){
      return 0;
    }
    int left = Math.max(0, getMax(root.left));
    int right = Math.max(0, getMax(root.right));
    max = Math.max(max, Math.max(root.val + Math.max(left, right), root.val + left + right));
    return root.val + Math.max(left, right);
  }


  /**
   * 剑指 Offer II 042. 最近请求次数
   */
  class RecentCounter {
    private Deque<Integer> deque;

    public RecentCounter() {
      this.deque = new LinkedList<>();
    }

    public int ping(int t) {
      while(!deque.isEmpty() && deque.peekFirst() + 3000 < t){
        deque.pollFirst();
      }
      deque.offerLast(t);
      return deque.size();

    }
  }

  /**
   * 剑指 Offer II 044. 二叉树每层的最大值
   */
  public List<Integer> largestValues(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if(root == null){
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()){
      int size = queue.size();
      int max = Integer.MIN_VALUE;
      for (int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        max = Math.max(max, node.val);
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
      }
      res.add(max);
    }
    return res;
  }

  /**
   * 剑指 Offer II 045. 二叉树最底层最左边的值
   */
  //bfs
  public int findBottomLeftValue(TreeNode root) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    int result = root.val;
    while(!queue.isEmpty()){
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        if(i == 0) result = node.val;
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
      }
    }
    return result;
  }
  //dfs
  int result = 0;
  int d = -1;
  public int findBottomLeftValue1(TreeNode root) {
    dfs3(root, 0);
    return result;

  }
  private void dfs3(TreeNode root, int depth){
    if(root == null) return;
    dfs3(root.left, depth+1);
    if(depth > d){
      d = depth;
      result = root.val;
    }
    dfs3(root.right, depth+1);
  }

  /**
   * 剑指 Offer II 046. 二叉树的右侧视图
   */
  public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if(root == null){
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while(!queue.isEmpty()){
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
        if(i == size-1){
          res.add(node.val);
        }
      }
    }
    return res;
  }
  //dfs 根右左
  List<Integer> res4 = new ArrayList<>();
  public List<Integer> rightSideView2(TreeNode root) {
    dfs4(root, 0);
    return res4;
  }
  private void dfs4(TreeNode root, int depth){
    if(root == null){
      return;
    }
    if(depth == res4.size()){
      res4.add(root.val);
    }
    dfs4(root.right, depth+1);
    dfs4(root.left,depth+1);
  }

  /**
   * 剑指 Offer II 047. 二叉树剪枝
   */
  public TreeNode pruneTree(TreeNode root) {
    if(root == null){
      return null;
    } //循环结束
    root.left = pruneTree(root.left);
    root.right = pruneTree(root.right);
    if(root.val == 0 && root.left == null && root.right == null) return null; //循环开始
    return root;
  }

  /**
   * 剑指 Offer II 049. 从根节点到叶节点的路径数字之和
   */
  int res5 = 0;
  public int sumNumbers(TreeNode root) {
    dfs5(root, 0);
    return res5;
  }
  private void dfs5(TreeNode root, int k){
    if(root == null){
      return;
    }
    if(root.left == null && root.right == null){
      res5 += k * 10 + root.val;
      return;
    }
    dfs5(root.left, k*10+root.val);
    dfs5(root.right, k*10+root.val);
  }

  /**
   *剑指 Offer II 050. 向下的路径节点之和
   * todo 前缀和+回溯理解
   */
  int res6 = 0;
  int targetSum;
  Map<Integer, Integer> hashmap = new HashMap<>(); //之前sum出现的次数
  public int pathSum1(TreeNode root, int targetSum) {
    this.targetSum = targetSum;
    hashmap.put(0, 1);
    dfs6(root, 0);
    return res6;
  }
  private void dfs6(TreeNode root, int sum){
    if(root == null){
      return;
    }
    sum += root.val;
    res6 += hashmap.getOrDefault(sum-targetSum,0);
    hashmap.put(sum, hashmap.getOrDefault(sum, 0)+1);
    dfs6(root.left, sum);
    dfs6(root.right, sum);
    hashmap.put(sum,hashmap.get(sum) - 1);
  }

  /**
   * 剑指 Offer II 051. 节点之和最大的路径
   */
  int res7 =  Integer.MIN_VALUE;;
  public int maxPathSum1(TreeNode root) {
    dfs7(root);
    return res7;
  }
  private int dfs7(TreeNode root){
    if(root == null){
      return 0;
    }
    int left = dfs7(root.left);
    int right = dfs7(root.right);
    res7 = Math.max(res7, left+right+root.val);
    return Math.max(0, Math.max(left, right)+root.val);

  }

  /**
   * 剑指 Offer II 052. 展平二叉搜索树
   */
  TreeNode preHead = new TreeNode(-1);
  TreeNode curNode = preHead;
  public TreeNode increasingBST(TreeNode root) {
    inOrder(root);
    return preHead.right;

  }
  private void inOrder(TreeNode root){
    if(root == null) return;
    inOrder(root.left);
    curNode.right = root;
    curNode = curNode.right;
    curNode.left = null;
    inOrder(root.right);
  }

  /**
   * 剑指 Offer II 053. 二叉搜索树中的中序后继
   * 递归
   */
  public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    if(root == null) return null;
    if(root.val > p.val){
      TreeNode n = inorderSuccessor(root.left, p);
      return  n==null?root:n;
    } else {
      return inorderSuccessor(root.right,p);
    }

  }

  /**
   * 剑指 Offer II 054. 所有大于等于节点的值之和
   */
  int sum = 0;
  public TreeNode convertBST(TreeNode root) {
    inOrder1(root);
    return root;
  }
  private void inOrder1(TreeNode root){
    if(root == null){
      return;
    }
    inOrder1(root.right);
    sum += root.val;
    root.val = sum;
    inOrder1(root.left);
  }

  /**
   * 剑指 Offer II 055. 二叉搜索树迭代器
   */
  class BSTIterator {
    private Queue<Integer> list = new LinkedList<>();

    public BSTIterator(TreeNode root) {
      inOrder2(root);
    }

    public int next() {
      int next = list.poll();
      return next;

    }

    public boolean hasNext() {
      return !list.isEmpty();

    }
    private void inOrder2(TreeNode root){
      if(root == null){
        return;
      }
      inOrder2(root.left);
      list.add(root.val);
      inOrder2(root.right);
    }
  }

  /**
   * 剑指 Offer II 056. 二叉搜索树中两个节点之和
   */
  List<Integer> list = new ArrayList<>();
  public boolean findTarget(TreeNode root, int k) {
    inOrder3(root);
    int l = 0;
    int r = list.size() - 1;
    while(l < r){
     if(list.get(l) + list.get(r) == k){
       return true;
     } else if(list.get(l) + list.get(r) < k){
       l++;
     } else{
       r--;
     }
    }
    return false;

  }
  private void inOrder3(TreeNode root){
    if(root == null){
      return;
    }
    inOrder3(root.left);
    list.add(root.val);
    inOrder3(root.right);
  }

  /**
   * 剑指 Offer II 057. 值和下标之差都在给定的范围内
   */
  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
    if(nums == null || nums.length == 0){
      return false;
    }
    int len = nums.length;
    for (int i = 0; i < len-k; i++) {
      for(int j = 1; j<=k; j++){
        if(Math.abs(nums[i]- nums[i+j]) <= t){
          return true;
        }
      }
    }
    return false;
  }

  /**
   * 剑指 Offer II 058. 日程表
   */
  class MyCalendar {
    List<int[]> list;


    public MyCalendar() {
      list = new ArrayList<>();
    }

    public boolean book(int start, int end) {
      for (int[] l : list) {
        if(l[0]< end && l[1]>start ){
          return false;
        }
      }
      list.add(new int[]{start, end});
      return true;

    }
  }

  /**
   * 剑指 Offer II 059. 数据流的第 K 大数值
   */
  class KthLargest {
    private PriorityQueue<Integer> pq;
    private int K;
    public KthLargest(int k, int[] nums) {
      pq = new PriorityQueue<>();
      this.K = k;
      for (int num : nums) {
        if(pq.size() < k){
          pq.add(num);
        } else {
          if(num > pq.peek()){
            pq.poll();
            pq.add(num);
          }
        }
      }
    }

    public int add(int val) {
      if(pq.size() < K){
        pq.add(val);
      } else {
        if(val > pq.peek()){
          pq.poll();
          pq.add(val);
        }
      }
      return pq.peek();

    }
  }

  /**
   * 剑指 Offer II 060. 出现频率最高的 k 个数字
   */
  public int[] topKFrequent(int[] nums, int k) {
    HashMap<Integer,Integer> hashMap = new HashMap<>();
    for (int num : nums) {
      hashMap.put(num, hashMap.getOrDefault(num, 0)+1);
    }
    PriorityQueue<int[]> pq = new PriorityQueue<>(((o1, o2) -> (o1[1]-o2[1])));
    for (Map.Entry<Integer, Integer> entry : hashMap.entrySet()) {
      if(pq.size() < k){
        pq.add(new int[]{entry.getKey(), entry.getValue()});
      } else {
        if(entry.getValue() > pq.peek()[1]){
          pq.poll();
          pq.add(new int[]{entry.getKey(), entry.getValue()});
        }
      }
    }
    int[] res = new int[k];
    for (int i = 0; i < k; i++) {
      res[i] = pq.poll()[0];
    }
    return res;

  }

  /**
   * 剑指 Offer II 061. 和最小的 k 个数对
   */
  public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
    List<List<Integer>> res = new ArrayList<>();
    int len1 = nums1.length;
    int len2 = nums2.length;
    PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> o2[0]+o2[1]-o1[0]-o1[1]);
    for(int i = 0; i<Math.min(len1, k); i++){
      for(int j = 0; j<Math.min(len2,k); j++){
        if(pq.size()<k){
          pq.add(new int[]{nums1[i], nums2[j]});
        } else {
          if(nums1[i] + nums2[j] < pq.peek()[0]+pq.peek()[1]){
            pq.poll();
            pq.add(new int[]{nums1[i],nums2[j]});
          }
        }
      }
    }
    while (!pq.isEmpty()) {
      List<Integer> list = new ArrayList<>();
      int[] peek = pq.poll();
      list.add(peek[0]);
      list.add(peek[1]);
      res.add(list);
    }
    return res;
  }

  /**
   * 剑指 Offer II 068. 查找插入位置
   */
  public int searchInsert(int[] nums, int target) {
    if(nums == null || nums.length == 0){
      return -1;
    }
    int len = nums.length;
    int l = 0;
    int r = len - 1;
    while (l<=r){
      int mid = l + (r-l)/2;
      if(nums[mid] == target){
        return mid;
      } else if(nums[mid] > target){
        r = mid;
      } else{
        l = mid;
      }
    }
    return l;

  }

  /**
   * 剑指 Offer II 069. 山峰数组的顶部
   */
  public int peakIndexInMountainArray(int[] arr) {
    if(arr==null || arr.length == 0){
      return -1;
    }
    int l = 0;
    int r = arr.length - 1;
    while(l<r){
      int mid = l+(r-l)/2;
      if(arr[mid] < arr[mid+1]) l = mid+1;
      else r = mid;
    }
    return l;
  }

  /**
   * 剑指 Offer II 070. 排序数组中只出现一次的数字
   */
  public int singleNonDuplicate(int[] nums) {
    if(nums == null || nums.length == 0){
      return -1;
    }
    int res = 0;
    for (int num : nums) {
      res ^= num;
    }
    return  res;
  }

  /**
   *剑指 Offer II 071. 按权重生成随机数
   */
  class Solution {
    int[] w;
    public Solution(int[] w) {
      for (int i = 1; i < w.length; i++) {
        w[i] += w[i-1];
      }
      this.w = w; //变成前缀和数组，表示每个索引的位置出现的概率
    }

    public int pickIndex() {
      int t = new Random().nextInt(w[w.length - 1]) +1;
      int l = 0;
      int r = w.length ;
      while(l <r){
        int mid = l + (r-l)/2;
        if(w[mid] >= t){
          r = mid;
        } else {
          l = mid + 1;
        }
      }
      return r;
    }
  }

  /**
   * 剑指 Offer II 072. 求平方根
   */
  public int mySqrt(int x) {
    if(x <= 0){
      return 0;
    }
    int l = 0;
    int r = x;
    int res = -1;
    while (l<=r){
      int mid = l+(r-l)/2;
      if((long)mid * mid > x){
        r = mid - 1;
      } else {
        l = mid + 1;
        res = mid;
      }
    }
    return res;

  }

  /**
   *剑指 Offer II 073. 狒狒吃香蕉
   */

  /**
   *剑指 Offer II 074. 合并区间
   */
  public int[][] merge(int[][] intervals) {
    if(intervals == null || intervals.length == 0){
      return new int[0][0];
    }
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        return o1[0] - o2[0];
      }
    });
    List<int[]> list = new ArrayList<>();
    int len = intervals.length;
    int preLeft = intervals[0][0];
    int preRight = intervals[0][1];
    list.add(new int[]{preLeft, preRight});
    for (int i = 1; i < len; i++) {
      preLeft = list.get(list.size() -1)[0];
      preRight = list.get(list.size() -1)[1];
      int left = intervals[i][0];
      int right = intervals[i][1];
      if(left > preRight){
        list.add(new int[]{left, right});
      } else {
        preRight = Math.max(preRight, right);
        list.remove(list.size()-1);
        list.add(new int[]{preLeft, preRight});
      }
    }
    int[][] res = new int[list.size()][2];
    for (int i = 0; i < list.size(); i++) {
      res[i] = list.get(i);
    }
    return res;
  }

  /**
   * 剑指 Offer II 075. 数组相对排序
   */
  public int[] relativeSortArray(int[] arr1, int[] arr2) {
    int max = Integer.MIN_VALUE;
    for (int i : arr1) {
      max = Math.max(max, i);
    }
    int[] cnt = new int[max+1];
    for (int i : arr1) {
      cnt[i]++;
    }

    int[] res = new int[arr1.length];
    int idx = 0;
    for (int num : arr2) {
      for(int i = 0; i<cnt[num]; i++){
        res[idx]= num;
        idx++;
      }
      cnt[num] = 0;
    }
    for (int i = 0; i < cnt.length; i++) {
      if(cnt[i] >0){
        for (int j = 0; j < cnt.length; j++) {
          res[idx] =i;
          idx ++;
        }
      }
    }
    return res;
  }

  /**
   *剑指 Offer II 076. 数组中的第 k 大的数字
   */
  public int findKthLargest(int[] nums, int k) {
    if(nums == null || nums.length == 0 || k < 0 || k>nums.length){
      return 0;
    }
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    for (int i = 0; i < nums.length; i++) {
      if(pq.size() < k){
        pq.add(nums[i]);
      } else {
        if(nums[i] >= pq.peek()){
          pq.poll();
          pq.add(nums[i]);
        }
      }
    }
    return pq.peek();
  }

  /**
   * 剑指 Offer II 077. 链表排序
   *
   */
  public ListNode sortList(ListNode head) {
    return merge(head);

  }
  private ListNode merge(ListNode head){
    if(head == null || head.next == null){
      return head;
    }
    ListNode slow = head;
    ListNode fast = head.next.next;
    while(fast != null && fast.next != null){
      slow = slow.next;
      fast = fast.next.next;
    }
    ListNode head1 = merge(slow.next);
    slow.next = null;
    //先链表变成有序的的，可以分治，变成单个元素
    ListNode head2 = merge(head);
    return mergeSort(head1, head2);
  }
  //只是合并两个有序的链表
  private ListNode mergeSort(ListNode head1, ListNode head2){
    if(head1 == null){
      return head2;
    }
    if(head2 == null){
      return head1;
    }
    ListNode preHead = new ListNode(-1);
    ListNode cur = preHead;
    while(head1 != null && head2 != null){
      if(head1.val > head2.val){
        cur.next = head2;
        head2 = head2.next;
      } else{
        cur.next = head1;
        head1 = head1.next;
      }
      cur = cur.next;
    }
    cur.next = head1 == null?head2:head1;
    return preHead.next;
  }

  /**
   *剑指 Offer II 078. 合并排序链表
   */
  public ListNode mergeKLists(ListNode[] lists) {
    if(lists == null || lists.length == 0){
      return null;
    }
    int len = lists.length;
    ListNode l1 = lists[0];
    for (int i = 1; i < len; i++) {
      ListNode l2 = lists[i];
      l1 = merge2Lists(l1, l2);
    }
    return l1;

  }
  private ListNode merge2Lists(ListNode l1, ListNode l2){
    if(l1 == null) return l2;
    if(l2 == null) return l1;
    if(l1.val < l2.val){
      l1.next = merge2Lists(l1.next, l2);
      return l1;
    } else{
      l2.next = merge2Lists(l2.next, l1);
      return  l2;
    }
  }

  /**
   * 剑指 Offer II 079. 所有子集
   */
  List<List<Integer>> res8 = new ArrayList<>();
  public List<List<Integer>> subsets(int[] nums) {
    List<Integer> list = new ArrayList<>();
    dfs8(list,0, nums);
    return res8;
  }
  private void dfs8(List<Integer> list, int idx, int[] nums){
    res8.add(new ArrayList<>(list));
    for (int i = idx; i < nums.length; i++) {
      list.add(nums[i]);
      dfs8(list,i+1,nums);
      list.remove(list.size() - 1);
    }
  }

  /**
   * 剑指 Offer II 080. 含有 k 个元素的组合
   */
  List<List<Integer>> res9 = new ArrayList<>();
  public List<List<Integer>> combine(int n, int k) {
    List<Integer> list = new ArrayList<>();
    dfs9(n,k,1,list);
    return res9;

  }
  private void dfs9(int n, int k, int idx, List<Integer> list){
    if(list.size() == k){
      res9.add(new ArrayList<>(list));
      return;
    }
    for(int i = idx; i<=n; i++){
      list.add(i);
      dfs9(n,k,i+1, list);
      list.remove(list.size() - 1);
    }
  }

  /**
   *剑指 Offer II 081. 允许重复选择元素的组合
   */
  List<List<Integer>> res10 = new ArrayList<>();
  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<Integer> list = new ArrayList<>();
    dfs10(list, candidates, target, 0);
    return res10;
  }
  private void dfs10(List<Integer> list, int[] candidates, int target, int idx){
    if(target < 0){
      return;
    } else if(target == 0){
      res10.add(new ArrayList<>(list));
      return;
    }
    for (int i = idx; i < candidates.length; i++) {
      list.add(candidates[i]);
      dfs10(list, candidates, target - candidates[i], i);
      list.remove(list.size() - 1);
    }
  }

  /**
   * 剑指 Offer II 082. 含有重复元素集合的组合
   */
  List<List<Integer>> res11 = new ArrayList<>();
  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<Integer> list = new ArrayList<>();
    Arrays.sort(candidates);
    dfs11(list, candidates, target, 0);
    return res11;
  }
  private void dfs11(List<Integer> list, int[] candidates, int target, int idx){
    if(target < 0){
      return;
    }
    if(target == 0){
      res11.add(new ArrayList<>(list));
      System.out.println(list);
      return;
    }
    for (int i = idx; i < candidates.length; i++) {
      if(i > idx && candidates[i] == candidates[i-1]) continue; //i>idx 在同一层里有重复的话，只选一个就
      list.add(candidates[i]);
      dfs11(list, candidates, target - candidates[i],i+1);
      list.remove(list.size() - 1);
    }
  }

  /**
   * 剑指 Offer II 083. 没有重复元素集合的全排列
   */
  List<List<Integer>> res12 = new ArrayList<>();
  public List<List<Integer>> permute(int[] nums) {
    Arrays.sort(nums);
    boolean[] visited = new boolean[nums.length];
    List<Integer> list = new ArrayList<>();
    dfs12(nums,visited, list);
    return res12;
  }
  private void dfs12(int[] nums, boolean[] visited, List<Integer> list){
    if(list.size() == nums.length){
      res12.add(new ArrayList<>(list));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if(visited[i]) continue;
      list.add(nums[i]);
      visited[i] = true;
      dfs12(nums, visited, list);
      visited[i] = false;
      list.remove(list.size() - 1);
    }

  }

  /**
   * 剑指 Offer II 084. 含有重复元素集合的全排列
   */
  List<List<Integer>> res13 = new ArrayList<>();
  public List<List<Integer>> permuteUnique(int[] nums) {
    List<Integer> list = new ArrayList<>();
    boolean[] visited = new boolean[nums.length];
    Arrays.sort(nums);
    dfs13(list, nums, visited);
    return res13;

  }
  private void dfs13(List<Integer> list, int[] nums, boolean[] visited){
    if(list.size() == nums.length){
      res13.add(new ArrayList<>(list));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if(visited[i]) continue;
      if(i > 0 && nums[i] == nums[i-1] && visited[i-1]){
        continue;
      }
      list.add(nums[i]);
      visited[i] = true;
      dfs13(list, nums, visited);
      visited[i] = false;
      list.remove(list.size() - 1);
    }
  }

  /**
   * 剑指 Offer II 085. 生成匹配的括号
   */
  public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    dfs14(res, "", 0,0, n);
    return res;

  }
  private void dfs14(List<String> res, String s, int l, int r, int n){
    if(s.length() == 2*n){
      res.add(s);
      return;
    }
    if(l < n){ //先将左括号排好
      dfs14(res, s +"(", l+1, r, n);
    }
    if(l > r){
      dfs14(res, s+")", l, r+1, n);
    }
  }

  /**
   * 剑指 Offer II 086. 分割回文子字符串
   */
  List<List<String>> res15 = new ArrayList<>();
  public String[][] partition(String s) {
    boolean[][] dp = new boolean[s.length()][s.length()];
    for (int i = 0; i < s.length(); i++) {
      dp[i][i] = true;
    }
    for(int len = 2; len<=s.length(); len++){
      for(int j = 0; j + len - 1 < s.length(); j++){
        int left = j;
        int right = j+len-1;
        if(len == 2){
          dp[left][right] = s.charAt(left) == s.charAt(right);
        } else {
          dp[left][right] = s.charAt(left) == s.charAt(right) && dp[left+1][right-1];
        }
      }
    }
    List<String> list = new ArrayList<>();
    System.out.println(dp);
    dfs15(list,s, 0,dp);
    String[][] res = new String[res15.size()][];
    for (int i = 0; i < res15.size(); i++) {
      res[i] = res15.get(i).toArray(new String[res15.get(i).size()]);
    }
    return res;
  }
  private void dfs15(List<String> list, String s, int idx, boolean[][] dp){
    if(idx == s.length()){
      res15.add(new ArrayList<>(list));
      return;
    }
    for (int i = idx; i < s.length(); i++) {
      if(dp[idx][i]){
        list.add(s.substring(idx,i+1));
        dfs15(list, s, i+1,dp);
        list.remove(list.size() - 1);
      }
    }

  }

  /**
   *剑指 Offer II 087. 复原 IP
   */
  List<String> res16 = new ArrayList<>();
  public List<String> restoreIpAddresses(String s) {
    List<String> path = new ArrayList<>();
    dfs16(s, 0, path);
    return res16;
  }
  private void dfs16(String s, int idx, List<String> path){
    if(path.size() == 4){
      if(idx == s.length()){
        String cur = String.join(".",path);
        res16.add(cur);
      }
      return;
    }
    if(idx == s.length()){
      return;
    }
    if(s.charAt(idx) == '0'){
      path.add("0");
      dfs16(s, idx+1, path);
      path.remove(path.size() - 1);
    } else {
      for (int i = idx; i < s.length(); i++) {
        int num = Integer.valueOf(s.substring(idx, i+1));
        if(num >= 0 && num<=255){
          path.add(s.substring(idx, i+1));
          dfs16(s, i+1, path);
          path.remove(path.size() - 1);
        } else{
          break;
        }
      }
    }

  }

  /**
   *剑指 Offer II 088. 爬楼梯的最少成本
   * dp[i] = Math.min(dp[i-1] + cost[i-1], dp[i-2]+cost[i-2]);
   * */
  public int minCostClimbingStairs(int[] cost) {
    if(cost == null || cost.length < 2){
      return 0;
    }
    int[] dp = new int[cost.length + 1];
    dp[0] = 0;
    dp[1] = 0;
    for (int i = 2; i <= cost.length; i++) {
      dp[i] = Math.min(dp[i-1]+cost[i], dp[i-2]+cost[i]);
    }
    return dp[cost.length];
  }

  /**
   *剑指 Offer II 089. 房屋偷盗
   * dp[i]表示偷盗到第i间时的金额
   * 对第K间房屋是否偷盗 dp[k] = max(dp[k-2] + nums[k], dp[k-1])
   */
  public int rob(int[] nums) {
    if(nums == null || nums.length == 0){
      return 0;
    }
    if(nums.length == 1){
      return nums[0];
    }
    int len = nums.length;
    int[] dp = new int[len];
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);
    for (int i = 2; i < len; i++) {
      dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
    }
    return dp[len - 1];
  }

  /**
   *剑指 Offer II 090. 环形房屋偷盗
   */
  public int rob2(int[] nums) {
    if(nums == null || nums.length == 0){
      return 0;
    }
    if(nums.length == 1){
      return nums[0];
    }
    int len = nums.length;
    return Math.max(robb(nums, 0, len - 2), robb(nums, 1, len - 1));
  }
  private int robb(int[] nums, int start, int end){
    int pre1 = 0;
    int pre2 = 0;
    int cur = 0;
    for(int i = start; i<=end; i++){
      cur = Math.max(pre2 + nums[i], pre1);
      pre2 = pre1;
      pre1 = cur;
    }
    return pre1;
  }

  /**
   * 剑指 Offer II 091. 粉刷房子
   */
  public int minCost(int[][] costs) {
    if(costs == null || costs.length == 0){
      return 0;
    }
    int len = costs.length;
    if(costs.length==1)return Math.min(costs[0][0],Math.min(costs[0][1],costs[0][2]));
    int[][] dp = new int[len][3];
    dp[0][0] = costs[0][0];
    dp[0][1] = costs[0][1];
    dp[0][2] = costs[0][2];
    for (int i = 1; i < costs.length; i++) {
      dp[i][0] = costs[i][0] + Math.min(dp[i-1][1], dp[i-1][2]);
      dp[i][1] = costs[i][1] + Math.min(dp[i-1][0], dp[i-1][2]);
      dp[i][2] = costs[i][2] + Math.min(dp[i-2][0], dp[i-1][1]);
    }

    return Math.min(dp[costs.length-1][0],Math.min(dp[costs.length-1][1],dp[costs.length-1][2]));
  }

  /**
   *剑指 Offer II 092. 翻转字符
   * 找到左边有多少个1，和右边有多少个0，表示的就是违规的需要改的数目，相加最小的就可以
   */
  public int minFlipsMonoIncr(String s) {
    int[] one = new int[s.length()];
    int[] zero = new int[s.length()];
    int temp = 0;
    for (int i = 0; i < s.length(); i++) {
      one[i] = temp;
      if(s.charAt(i) == '1'){
        temp++;
      }
    }
    temp = 0;
    for(int i=s.length() - 1; i>=0; i--){
      zero[i] = temp;
      if(s.charAt(i) == '0'){
        temp++;
      }
    }
    int min = Integer.MAX_VALUE;
    for (int i = 0; i < s.length(); i++) {
      min = Math.min(min, zero[i]+one[i]);
    }
    return min;
  }

  /**
   *剑指 Offer II 093. 最长斐波那契数列
   */
  public int lenLongestFibSubseq(int[] arr) {
    if(arr == null || arr.length < 3){
      return 0;
    }
    Set<Integer> set = new HashSet<>();
    for (int i : arr) {
      set.add(i);
    }
    int res = 0;
    for (int i = 0; i < arr.length; i++) {
      for (int j = i+1; j < arr.length; j++) {
        int left = arr[i];
        int right = arr[j];
        int len = 2;
        while(set.contains(left+right)){
          int tmp = left+right;
          left = right;
          right = tmp;
          len++;
        }
        if(len > 2){
          res = Math.max(res, len);
        }

      }
    }
    return res;

  }

  /**
   *剑指 Offer II 095. 最长公共子序列
   * dp[i][j]表示s1的第i个下标和s2的第j个下标公共的子序列的长度
   * */
  public int longestCommonSubsequence(String text1, String text2) {
    int[][] dp = new int[text1.length()+1][text2.length()+1];
    for (int i = 1; i <= text1.length(); i++) {
      for (int j = 1; j <= text2.length(); j++) {
        if(text1.charAt(i-1) == text2.charAt(j-1)){
          dp[i][j] = dp[i-1][j-1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
        }
      }
    }
    return dp[text1.length()][text2.length()];
  }

  /**
   * 剑指 Offer II 096. 字符串交织
   * dp[i][j] 表示s3的i+j个字符是否能由s1的前i个和s2的前j个组合
   */
  public boolean isInterleave(String s1, String s2, String s3) {
    int l1 = s1.length();
    int l2 = s2.length();
    int l3 = s3.length();
    if(l2 + l1 != l3){
      return false;
    }
    boolean[][] dp = new boolean[l1+1][l2+1];
    dp[0][0] = true;
    for (int i = 0; i <=l1; i++) {
      for (int j = 0; j <= l2; j++) {
        if( i >= 1 &&s3.charAt(i+j-1) == s1.charAt(i-1)){
          dp[i][j] = dp[i-1][j] || dp[i][j];
        }
        if( j >=1 && s3.charAt(i+j-1) == s2.charAt(j-1)){
          dp[i][j] = dp[i][j] || dp[i][j-1];
        }
      }
    }
    return dp[l1][l2];
  }

  /**
   * 剑指 Offer II 098. 路径的数目
   */
  public int uniquePaths(int m, int n) {
    int[][] dp = new int[m+1][n+1];
    for (int i = 0; i <= m; i++) {
      dp[i][0] = 1;
    }
    for (int i = 0; i <= n; i++) {
      dp[0][i] = 1;
    }
    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        dp[i][j] = dp[i-1][j] + dp[i][j-1];
      }
    }
    return dp[m][n];
  }

  /**
   *剑指 Offer II 099. 最小路径之和
   */
  public int minPathSum(int[][] grid) {
    if(grid == null || grid.length == 0){
      return 0;
    }
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int[m][n];
    int res = Integer.MAX_VALUE;
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; i++) {
      dp[i][0] = dp[i-1][0] + grid[i][0];
    }
    for (int i = 1; i < n; i++) {
      dp[0][i] = dp[0][i-1]+grid[0][i];
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1])+grid[i][j];
      }
    }
    return dp[m-1][n-1];
  }

  /**
   *剑指 Offer II 100. 三角形中最小路径之和
   * dp[i][j] = min(dp[i-1][j] ,dp[i-1][j-1])+tr.get(i).get(j)
   */
  public int minimumTotal(List<List<Integer>> triangle) {
    int m = triangle.size();
    int[][] dp = new int[m][m];
    int pre = 0;
    dp[0][0] = triangle.get(0).get(0);
    for(int i=1;i<triangle.size();i++){
      dp[i][0] = dp[i-1][0]+triangle.get(i).get(0);
    }
    for (int i = 1; i < m; i++) {
      for(int j=1;j<triangle.get(i).size();j++){ //对每一个list中元素进行遍历
        if(j<triangle.get(i-1).size()){
          dp[i][j] = Math.min(dp[i-1][j-1],dp[i-1][j])+triangle.get(i).get(j);
        }else{
          dp[i][j] = dp[i-1][j-1]+triangle.get(i).get(j);
        }
      }

    }
    int min = Integer.MAX_VALUE;
    for(int i=0;i<triangle.size();i++){
      if(dp[triangle.size()-1][i]<min){
        min = dp[triangle.size()-1][i];
      }
    }
    return min;

  }

  /**
   * 剑指 Offer II 101. 分割等和子串
   * 01背包问题的变形 todo 01背包
   * dp[i][j] 表示i个元素中的中是否有元素的和能为j
   * dp[i][j] = dp[i][j-1]
   * */
  public boolean canPartition(int[] nums) {
    int sum = 0;
    for (int num : nums) {
      sum += num;
    }
    if(sum % 2  != 0){
      return false;
    }
    int target = sum/2;
    boolean[][] dp = new boolean[nums.length+1][target + 1];
    for(int i = 0; i<=nums.length; i++){
      dp[i][0] = true;
    }
    for(int i = 0; i<=target; i++){
      dp[0][i] = false;
    }
    for(int i = 1; i<=nums.length; i++){
      for(int j = 1; j<=target; j++){
        if(j >= nums[i-1]){ //如果target 比当前值大，则可以取当前值
          dp[i][j] = dp[i-1][j] || dp[i-1][j - nums[i-1]]; //如果前i-1个元素可以等于j-num[i-1],那么加上num[i-1]也可以
        } else {
          dp[i][j] = dp[i-1][j]; //如果前i-1个元素可以，那么不取第i个元素也可以
        }

      }
    }
    return dp[nums.length][target];
  }

  /**
   * 剑指 Offer II 102. 加减的目标值
   */
//  public int findTargetSumWays(int[] nums, int target) {
//
//  }

  public String solution(String S) {
    // write your code in Java SE 8
    int[] arr = new int[26];
    char[] chars = S.toCharArray();
    for (int i = 0; i < chars.length; i++) {
      if(chars[i] >= 65 && chars[i]<= 90){ // 说明是大写
        int idx = chars[i] - 'A';
        if(arr[idx] == 0){
          arr[idx] = 1;
        } else if(arr[idx] == -1){
          arr[idx] = 2;
        } else {
          continue;
        }
      } else if(chars[i] >= 97 && chars[i] <= 122){ // 说明是小写
        int idx = chars[i] - 'a';
        if(arr[idx] == 0){
          arr[idx] = -1;
        } else if(arr[idx] == 1){
          arr[idx] = 2;
        } else {
          continue;
        }
      }
    }
    for (int i = arr.length - 1; i >= 0; i--) {
      if(arr[i] == 2){
        int idx = 65 + i;
        char ch = (char) idx;
        return String.valueOf(ch);
      }
    }
    return "NO";
  }

  public int solution(int A, int B) {
    // write your code in Java SE 8
    int max = (A+B)/4;
    for (int i = max; i >= 1; i--) {
      if(A/i + B/i  >= 4){
        return i;
      }
    }
    return 0;
  }
  public  int solution(int[] A) {
    // write your code in Java SE 8
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < A.length; i++) {
      map.put(i, A[i]);
    }
    List<Map.Entry<Integer,Integer>> list = new ArrayList<Map.Entry<Integer,Integer>>(map.entrySet());
    Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
      @Override
      public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
        return (o1.getValue()).compareTo(o2.getValue());
      }
    });
    List<Integer> l = new ArrayList<>();
    for (Map.Entry<Integer, Integer> entry : list){
      l.add(entry.getKey());
    }
    boolean isFirst = true;
    boolean minExist = false;
    int res = 0;
    int min = 0;
    int max = 0;
    int sum = 0;
    for (int i = 0; i < l.size(); i++) {
      if(l.get(i) == min){
        minExist = true;
      }
      sum += l.get(i);
      max = Math.max(max, l.get(i));
      min = Math.min(min,l.get(i));
      if(isFirst && !minExist){
        continue;
      }
      if(isContinue(sum, max, min) || l.get(i) == max + 1 ){
        res += 1;
        sum = 0;
        min = max + 1;
        max = 0;
        isFirst = false;
      } else {
        continue;
      }
    }
    return res;
  }
  private  boolean isContinue(int sum, int max, int min){
    if((max + min)*(max-min+1) / 2 == sum ){
      return true;
    } else {
      return false;
    }
  }

  public static void main(String[] args) {
    int res = lengthOfLongestSubstring("abba");
    Offer2 offer2 = new Offer2();
    int[] a = new int[]{2,4,1,6,5,9,7};
    int[] b = new int[]{4,3,2,6,1};
    int[] c = new int[]{2,1,6,4,3,7};
    int r = offer2.solution(a);
//    offer2.partition("bb");
   System.out.println(r);

  }



}

