import java.util.*;

/**
 * User: gaohan
 * Date: 2021/1/31
 * Time: 10:52
 */
public class Offer {

  /****面试题3：数组中的重复元素
   * Arrays.sort 的时间复杂度为O(nlogn)
   * findRepeatNumber1: 暴力解法，返回的总是重复元素中最小元素，时间复杂度O(nlogn)
   * findRepeatNumber2：利用哈希表解决， 再定义一个大小相同的数组，时间复杂度为O(n),空间复杂度为O(n),拿空间换时间
   * findRepeatNumber3: 原地置换(原地哈序)，重排之后，如果没有重复元素，那么索引为i的元素也应该是i，如果索引为i的元素m != i,那么将i索引的元素与 m索引的元素进行交换，这样m索引处的元素就是m,
   * 交换的过程中，判断是否有重复得到元素,时间复杂度为O(n),空间复杂度为O(1)
   * findRepeatNumber4：在上述方法中，如果要求不改变原来的数组，并且只有一个重复元素时，可以用二分法，计算小于m的元素数目,复杂度O(n logn)
   * ****/
  public int findRepeatNumber1(int[] nums){
    if (nums == null || nums.length == 0){
      return -1;
    }
    Arrays.sort(nums);
    for(int i = 0; i<nums.length-1; i++){
      if(nums[i] == nums[i+1]){
        return nums[i];
      }
    }
    return -1;

  }
  public int findRepeatNumber2(int[] nums){
    if (nums == null || nums.length == 0){
      return -1;
    }
    int[] cnt= new int[nums.length];
    for(int i = 0; i<nums.length; i++){
      cnt[nums[i]] += 1;
      if(cnt[nums[i]] > 1){
        return nums[i];
      }
    }
    return -1;


  }
  //鸽巢原理，先将num[i]归位，如果发现将num[i]归位时，num[i]下标处得到元素和要归位的元素一样，说明存在重复元素
  // 一个萝卜一个坑
  public int findRepeatNumber3(int[] nums){
    int temp = 0;
    if (nums == null || nums.length == 0){
      return -1;
    }
    for(int i = 0; i<nums.length; i++){
      while(nums[i] != i){ // 只有本元素归位正确后，再进行下一个归位
        if(nums[i] == nums[nums[i]]){ //经过转换后，每个索引i对应的元素应该是i,如果此时相同后，说明num[i]这个元素在i位置和num[i]位置两个地方出现
          return nums[i];
        }
        temp = nums[i];
        nums[nums[i]] = nums[temp];
        nums[temp] = temp;

      }
    }
    return -1;
  }

  public int findRepeatNumber4(int[] nums){
    if (nums == null || nums.length == 0){
      return -1;
    }
    int l = 0;
    int r = nums.length -1;
    while(l < r){
      int m = l + (r-l)/2;
      int cnt = countRange(nums, m); //计算小于等于m的元素数量

      if(cnt > m ){ //如果比m小的元素大于m,说明重复元素在左侧
        r = m;
      } else{
        l = m + 1;
      }
    }
    return l;

  }
  private int countRange(int[] nums, int m){
    int count = 0;
    for(int n :nums){
      if(n <= m){
        count++;
      }
    }
    return count;
  }

  /****面试题4：二维数组中的查找
   *findNumberIn2DArrays1 暴力查找，O(m * n)
   * 考虑到递增的特征，以右上角的元素为基准，如果tar 比右上角的元素小，则可剔除该列，如果比右上角元素大，则剔除该行O(m+n)
   * ****/
  public boolean findNumberIn2DArrays1(int[][] nums, int target){
    if(nums == null|| nums.length == 0 || nums[0].length == 0){
      return false;
    }
    int row = nums.length;
    int col = nums[0].length;
    for(int i = 0; i<row; i++){
      for(int j = 0; j<col; j++){
        if(nums[i][j] == target){
          return true;
        }
      }
    }
    return false;
  }
  public boolean findNumberIn2DArrays2(int[][] nums, int target){
    if(nums == null|| nums.length == 0 || nums[0].length == 0){
      return false;
    }
    int row = nums.length;
    int col = nums[0].length;
    int i = 0;
    int j = col - 1;
    while(i < row && j >=0 ){
      if(target < nums[i][j] ){
        j--;
      } else if(target > nums[i][j]) {
        i++;
      } else {
        return true;
      }
    }
    return false;

  }

  /***面试题5：替换空格
   *O(n)
   * ****/
  public String replaceSpace(String str){
    StringBuilder sb =  new StringBuilder();
    for(char c: str.toCharArray()){
      if(c == ' '){
        sb.append('%').append('2').append('0');
      } else {
        sb.append(c);
      }
    }
    return sb.toString();

  }

  /***面试题6：从尾到头打印链表
   * printListNode1: 利用栈O（n）
   * printListNode2: 两次遍历链表，第一次遍历时不要修改head 的额=值，用一个headNode 来遍历
   * printListNode3：递归在本质上就是一个栈结构，所以可以使用递归的方法
   * ****/
  public class ListNode{
    int val;
    ListNode next;
    public ListNode(int x){val = x;};
  }
  public int[] printListNode(ListNode head){
    if(head == null){
      return new int[0];
    }
    Stack<Integer> stack = new Stack<>();
    while(head != null){
      stack.push(head.val);
      head = head.next;
    }
    int len = stack.size();
    int[] res = new int[len];
    for(int i = 0; i<len; i++){
      res[i] = stack.pop();
    }

    return res;

  }
  public int[] printListNode2(ListNode head){
    if(head == null){
      return new int[0];
    }
    int len = 0;
    ListNode headNode = head;

    while(headNode != null){
      len++;
      headNode  = headNode.next;
    }
    int[] res = new int[len];
    int index = 0;
    while(head != null){
       res[len - 1- index] = head.val;
       index ++;
       head = head.next;

    }
    return res;
  }
  List<Integer> list = new ArrayList<>();
  public int[] reversePrint(ListNode head){
    if(head == null){
      return new int[0];
    }
    recur(head);
    int[] res = new int[list.size()];
    for(int i = 0; i<res.length; i++){
      res[i] = list.get(i);
    }
    return res;

  }
  private void recur(ListNode head){
    if(head == null) return;
    recur(head.next);
    list.add(head.val);
  }

  /****面试题07：重建二叉树
   * 递归：通过根节点得到左子树的范围和右子树的范围, 更新左子树和右子树的数组索引值
   * *****/
  public class TreeNode{
    int val ;
    TreeNode left;
    TreeNode right;
    TreeNode next;
    TreeNode (int x){val = x;}
  }
  HashMap<Integer, Integer> inOrderIndex = new HashMap<>();
  public TreeNode buildTree(int[] preOrder,int[] inOrder){
    if(preOrder == null || preOrder.length == 0 || inOrder == null || inOrder.length == 0 || preOrder.length != inOrder.length){
      return null;
    }
    int n = preOrder.length;
    for(int i = 0; i<n; i++){
      inOrderIndex.put(inOrder[i], i);
    }
    return build(preOrder, inOrder, 0, n-1, 0, n-1);


  }
  private TreeNode build(int[] preOrder, int[] inOrder, int preLeft, int preRight, int inLeft, int inRight){
    if(preLeft > preRight){
      return null;
    } //递归终止的条件，说明是叶子结点
    int rootIndex = inOrderIndex.get(preOrder[preLeft]);
    int sizeLeft = rootIndex - inLeft; //
    TreeNode root = new TreeNode(preOrder[preLeft]); //前序遍历的数组的最左边的值为根
    root.left = build(preOrder, inOrder,preLeft+1,preLeft + sizeLeft , inLeft,rootIndex-1 );
    root.right =  build(preOrder, inOrder,preLeft+sizeLeft+1, preRight, rootIndex+1,inRight );
    return root;
  }

  /****面试题8： 二叉树的下一个节点：
   * 给出一个节点，寻找中序遍历序列中的下一个节点
   * ****/
  public TreeNode getNext(TreeNode node){
    if(node.right != null){
      TreeNode right = node.right;
      while(right.left != null){
        right = right.left;
      }
      return right;
    } else {
      while(node.next != null){
        TreeNode parent = node.next;
        if(parent.left == node){
          return parent;
        }
        node = node.next;
      }

    }
    return null;

  }

  /****面试题9 用两个栈实现队列
   * *****/
  public class CQueue{
    Stack<Integer> s1;
    Stack<Integer> s2;
    public CQueue(){
      s1 = new Stack<>();
      s2 = new Stack<>();

    }
    public void appendTail(int value){
      s1.push(value);

    }
    public int deleteHead(){
      if(s2.empty()){
        while(!s1.empty()){
          s2.push(s1.pop());
        }
        if(s2.isEmpty()){
          return -1;
        } else {
          return s2.pop();
        }
      }else {
        return s2.pop();
      }
      //每次添加和删除都得倒腾
//      if(s1.empty()){
//        return -1;
//      }
//      while(!s1.empty()){
//        s2.push(s1.pop());
//      }
//      int res = s2.pop();
//      while(!s2.empty()){
//        s1.push(s2.pop());
//      }
//      return res;

    }
  }

  /****225. 用队列实现栈
   * 用一个队列更好理解
   * ******/
  class MyStack {
    Queue<Integer> q1;
    Queue<Integer> q2;

    /** Initialize your data structure here. */
    public MyStack() {
      q1 = new LinkedList<>(); //输入队列
      q2 = new LinkedList<>();  //输出队列
    }

    /** Push element x onto stack. */
    public void push(int x) {
      q1.offer(x);
      //将输出队列中的元素全部转入输入队列中
      while(!q2.isEmpty()){
        q1.offer(q2.poll());
      }
      //交换两个队列，使输入队列在没有push的时候始终为空
      Queue a = q1;
      q1 = q2;
      q2 = a;

    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
      return q2.poll();

    }

    /** Get the top element. */
    public int top() {
      return q2.peek();

    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
      return  q2.isEmpty();

    }
  }

  /***面试题10 斐波那契数列
   * 直接递归
   * 循环，保留变量
   * 动态规划
   * ****/
  public int fib1(int n){
    if(n <= 0){
      return 0;
    }
    if(n == 1){
      return 1;
    }
    return fib1(n-1)+fib1(n-2);
  }
  public int fib2(int n){
    int[] res = {0,1};
    if(n <2){
      return res[n];
    }
    int fibNum1 = 0;
    int fibNum2 = 1;
    int fibN = 0;
    for(int i = 2; i<=n; i++){
      fibN = fibNum1 + fibNum2;
      // 会超过int 能表示的范围，所以要取余
      if (fibN >= 1000000007) {
        fibN -= 1000000007;
      };
      fibNum1 = fibNum2;
      fibNum2 = fibN;
    }
    return fibN;

  }
  public int fib3(int n){
    if(n <= 0){
      return 0;
    }
    if(n == 1){
      return 1;
    }
    int[] dp = new int[n+1];
    dp[0] = 0;
    dp[1] = 1;
    for(int i = 2; i<= n; i++){
      dp[i] = dp[i-1]+dp[i-2];
      if (dp[i]  >= 1000000007) {
        dp[i]  -= 1000000007;
      };
    }
    return dp[n];
  }

  /****面试题10：青蛙跳台阶
   * 动态规划：在某一处的台阶可以由上一台阶跳一步，也可由上二个台阶跳两步，dp[i] = dp[i-1] + dp[i-2];
   *
   * * ****/
  public int wayNums(int n){
    if (n == 0) {
      return 1;
    }
    if (n == 1) {
      return 1;
    }
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
      if (dp[i] >= 1000000007) {
        dp[i] -= 1000000007;
      }
    }

    return dp[n];
  }

  /**** 面试题11 旋转数组的最小数组
   * minNum1 暴力常规解法：O（n）的时间复杂度，找到一个元素的下一个元素比该元素下，则下一个元素就是转折点
   * 二分：考虑到已经排好序的特征，可以使用二分法，O(logn)
   * 可能会有重复的元素
   * 不能与左边的数相比，不能有效的减治
   * fixme 需要注意的细节较多哦
   * ****/
  public int minNum1(int[] nums){
    if(nums == null ||nums.length == 0){
      return -1;
    }
    for(int i = 0; i<nums.length-1; i++){
      if(nums[i] > nums[i+1]){
        return nums[i+1];
      }
    }
    return nums[0];
  }

  // 不能与左边的数进行比较，不能有效的减治
  public int minNum2(int[] nums){
    if(nums == null ||nums.length == 0){
      return -1;
    }
    int left = 0;
    int right = nums.length - 1;
    while (left < right){
      int mid = left + (right - left)/2;
      if(nums[mid] > nums[right]){ //说明mid 以及 mid 的左边一定不是最小元素，最小元素在右边
        left = mid + 1;
      } else if(nums[mid] < nums[right]){ //mid的右边是增长的， 右边一定没有最小元素，mid 有可能是
        right = mid;
      } else {
        right -= 1;  //此时无法判断，只能缩小范围排除right
      }
    }
    return nums[left];
  }

  /****面试题12 矩阵中的路径
   * 回溯
   * 注意index ++ 和index+1
   * 回溯终止的条件是，把字符串遍历完，看char 是否相等
   * *****/
  public boolean exist(char[][] board, String s){
    if(board == null || board.length == 0 || board[0].length == 0 || s.isEmpty()){
      return false;
    }
    int row = board.length;
    int col = board[0].length;
    boolean[][] visted = new boolean[row][col];
    for(int i = 0; i<row; i++){
      for(int j = 0; j<col; j++){
        if(dfs12(board, i, j, row, col, visted, s, 0)){
          return true;
        }
      }
    }
    return false;

  }
  int[][] forward = {{1,0},{-1, 0},{0,1},{0,-1}};
  private boolean dfs12(char[][] board, int r, int c, int row, int col, boolean[][] visited, String s, int index){

    // 如果搜索的点与字符串中的第k个字符不相同
    if (board[r][c] != s.charAt(index)) {
      return false;
    }
    // 整个字符串都已经找到了，返回true
    else if (index == s.length() - 1) {
      return true;
    }


    if(board[r][c] == s.charAt(index)){
      visited[r][c] = true;
      for(int i = 0; i<4; i++){
        int newX = r + forward[i][0];
        int newY = c + forward[i][1];
        if(newX >=0 && newX <row && newY >=0 && newY <col && !visited[newX][newY]){ //注意判断的顺序，先判断是否在边界内
          if(dfs12(board, newX, newY, row, col, visited, s, index+1)){ //注意index ++ 和index+1 的区别
            return true;
          }
        }
      }
      visited[r][c] = false;
    }
    return false;

  }

  /***面试题13 机器人的运动范围
   * ****/
  public int movingCount(int m, int n, int k){
    if(m == 0 && n == 0){
      return 0;
    }
    boolean[][] visited = new boolean[m][n];
    int count = 0;

    return dfs13(0,0, m, n, visited, k);
  }
  private int dfs13(int i, int j, int m, int n, boolean[][] visited, int k){
    if(!check(i, j, k) || i <0 || j<0 || i>=m || j >=n || visited[i][j]){
      return 0;
    } // 更优雅的结束回溯的判断  todo

    visited[i][j] = true;
    int res = 1;
    for(int l = 0; l<4; l++){
      int newI = i + forward[l][0];
      int newJ = j + forward[l][1];
      res +=  dfs13(newI, newJ, m, n, visited, k);
    }
    return res;

  }

  private boolean check(int i, int j, int k){
    int i1 = i/10;
    int i2 = i%10;
    int j1 = j/10;
    int j2 = j%10;
    if(i1+i2+j1+j2 > k ){
      return false;
    } else {
      return true;
    }
  }

  /****面试题14：剪绳子1 动态规划
   * dp[n] 表示把绳子剪开后的各段乘积的最大值
   * dp[n] = max(dp[n-i] * dp[i]) 0<i<n
   * 注意 i = 1 2 3 时的特殊情况，分为1，2，3 的情况时，说明已经分过了，1， 2， 3 这部分再分的话不能比他本身没分割的要小
   * 或者  dp[i] = Math.max(dp[i], Math.max(j * dp[i-j], j * (i - j)))
   * ****/
  public int cuttingRop(int n){
    if(n <= 2){
      return 1;
    } else if(n == 3){
      return 2;
    }
    int[] dp = new int[n+1];
    //真正的dp[1] = 0,dp[2] = 1,dp[3] = 2, 当n=4时，4=2+2 2*2=4 而dp[2]=1是不对
    //也就是说当n=1/2/3时，分割后反而比没分割的值要小，当大问题用到dp[j]时，说明已经分成了一个j一个i-j，这两部分又可以再分，但是再分不能比他本身没分割的要小，如果分了更小还不如不分
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 3;

    for(int i = 4; i<=n; i++){
      for(int j = 1; j<= i/2; j++){ // 取i/2 减少循环次数
        dp[i] = Math.max(dp[i], dp[j] * dp[i - j]);
      }
    }
    return dp[n];
  }

  /****面试题14：剪绳子1 动态规划
   *贪心算法：当 n >=5 时，2（n-2）>n ,3(n-3) > n
   * 说明剩余的绳子长度大于等于5时，剪成长度为2 或3 的绳子段
   * 3(n-3) > 2（n-2）所以要尽可能的剪成3
   */
  public int cuttingRop1(int n){
    if(n <= 2){
      return 1;
    } else if(n == 3){
      return 2;
    } else if(n == 4){
      return 4;
    }

    int timesOf3 = n/3;
//    if(n - timesOf3 * 3 == 1){
//      timesOf3 -= 1;
//    }
//    int timesOf2 = (n - timesOf3 * 3)/2;
//    double res =  Math.pow(2, timesOf2) * Math.pow(3, timesOf3);;
//    if (res  >= 1000000007) {
//      res  -= 1000000007;
//    };
//
//    return  (int) res ;

    long res = 1;
    while(n > 4){
      res *= 3;
      res %= 1000000007;
      n -= 3;
    }
    return (int) (res*n%1000000007);

  }

  /****面试题15 二进制中1 的个数
   *考虑输入的数，是正数还是负数 或0
   * 无符号数，可以用常规解法
   * 右移比除法要快
   * 思路：n & (n-1) 得到的数比n 少一个1，相当于把整数二进制中最右边的1 变成0
   *
   *
   * ****/
   public int numberOf1(int n){
     //return Integer.bitCount(n);
     int cnt = 0;
     while(n != 0){
       cnt++;
       n = n &(n-1);
     }
     //常规解法，只能用于无符号数，负数会死循环
//     while (n != 0) {
//       cnt += n&1; // n的最低位与1 做与运算，如果是1 ，则加一
//       n = n >>> 1;  // 无符号右移1位，
//     }


     return cnt;

   }

   /***面试题16：数值的额整数次方
    * 本题考的是：思考的全面性，要考虑x 和 n 的取值，对x==0 && n<0的异常情况，利用全局变量来表示是否出现异常
    * 快速做乘方，提高效率
    * ****/
   boolean invalidInput = false;
   public double myPow(double x, int n) {
     invalidInput = false;

     if(x == 0.0 && n<0){
       invalidInput = true;
       return 0.0;
     }
     double res = mypower(x, Math.abs(n));

     if(n < 0){
       res = 1.0/res;
     }
     return res;

   }
   // 该函数的n一定大于等于0,0的0次方没有意义，所以也可以输出1
   private double mypower(double x, int n){
     if(n == 0){
       return 1;
     }
     if(n == 1){
       return x;
     }
     double res = mypower(x, n/2);
     res *= res;
     if(n % 2 != 0){
       res = res * x;
     }

     return res;

   }

   /****面试题17 打印从1到最大的n位数
    * fixme 问清楚数字的范围
    * 大数问题： 用字符串String，全排列问题 ----- todo
    * ****/
   public int[] printNumbers(int n) {
     if(n <= 0){
       return new int[0];
     }
     int len = (int)Math.pow(10, n);
     int[] res = new int[len - 1];
     for(int i = 0; i<len-1; i++){
       res[i] = i+1;
     }
     return res;


   }
   // 全排列
  public void print1ToMaxOfNDigits(int n) {
    if (n <= 0)
      return;
    char[] number = new char[n];
    print1ToMaxOfNDigits(number, 0);
  }

  private void print1ToMaxOfNDigits(char[] number, int digit) {
    if (digit == number.length) {
      printNumber(number);
      return;
    }
    for (int i = 0; i < 10; i++) {
      number[digit] = (char) (i + '0');
      print1ToMaxOfNDigits(number, digit + 1);
    }
  }

  private void printNumber(char[] number) {
    int index = 0;
    // 找到字符串中第一个值不为0元素，前边的0不输出
    while (index < number.length && number[index] == '0')
      index++;

    while (index < number.length)
      System.out.print(number[index++]);

    System.out.println();
  }

  /****面试题18 删除链表的节点
   * 注意：删除头节点，尾节点，为空的链表
   * ****/
  //普通方法：时间复杂度O（n）
  public ListNode deleteNode(ListNode head, int val){
    if(head == null){
      return head;
    } // 边界条件
    if(head.val == val){
      return head.next;
    } //边界条件
    ListNode prehead = head;
    while(prehead != null && prehead.next != null){ //得是两个都不为空
      if(prehead.next.val == val){
        prehead.next = prehead.next.next;
      }
      prehead = prehead.next;
    }
    return head;
  }

  // 在O（1）时间内删除链表节点, 不能确定要删除的元素是否一定存在
  public ListNode deleteNode1(ListNode head, ListNode tobeDelete){
    if(head == null || tobeDelete == null){
      return null;
    }
    if(tobeDelete.next != null){ //O(n-1)
      ListNode next = tobeDelete.next;
      tobeDelete.val = next.val;
      tobeDelete.next = next.next;

    } else{ //如果是尾部节点，需要先找出尾部节点的前序节点 O(n)
      ListNode cur = head;
      while(cur.next != tobeDelete){
        cur = cur.next;
      }
      cur.next = null;
    }
    return head;

  }

  /****面试题18-2 删除排序链表中重复的节点
   * 不是去重，而是删掉所有重复元素,采用递归的方法
   * ****/
  public ListNode deleteDuplication(ListNode pHead) {
    if(pHead == null ||pHead.next == null){
      return pHead;
    }
    ListNode next = pHead.next;
    if(pHead.val == next.val){ //头节点的两个相同时
      while(next != null && pHead.val == next.val){
        next = next.next;
      }
      return deleteDuplication(next);
    } else {
      pHead.next = deleteDuplication(pHead.next);
      return pHead;
    }

  }

  /****面试题19 正则表达式匹配
   * dp[i][j] 表示s的前i个字符和p的前j个字符是否对的上 todo
   * * *****/
//  public boolean isMatch(String s, String p){
//    char[] str = s.toCharArray();
//    char[] pat = p.toCharArray();
//    int m = str.length;
//    int n = pat.length;
//    boolean[][] dp = new boolean[m+1][n+1];
//
//    dp[0][0] = true;
//    for(int i = 1; i<n; i++){
//      if()
//    }
//
//  }

  /***面试题20：表示数值的字符串
   * ****/
//  public boolean isNumber(String s) {
//    s.
//
//  }

  /****面试题21：调整数组的顺序使奇数位于偶数前边
   * 思路1：先计算出奇数的数量，根据索引，依次填充，时间和空间复杂度都是O（n）,优点是不改变原数组
   * 思路2：双指针
   * ******/
  public int[] exchange(int[] nums){
    int len = nums.length;
    int oddCnt = 0;
    for(int n: nums){
      if(n%2 == 1){
        oddCnt += 1;
      }
    }
    int i = 0;
    int j = oddCnt;
    int[] res = new int[len];
    for(int n : nums){
      if(n % 2 != 0){
        res[i++] = n;
      } else {
        res[oddCnt++] = n;
      }
    }
    return res;
  }

  //时间O(n) 空间O(1)
  public int[] exchange1(int[] nums) {
    int len = nums.length;
    if(len < 2){
      return nums;
    }
    int left = 0;
    int right = len - 1;
    while (left <right){
      while(left < right && nums[left] % 2 != 0){
        left++;
      }

      while(left < right && nums[right] % 2 == 0){
        right --;
      }

      int temp = nums[left];
      nums[left] = nums[right];
      nums[right] = temp;
    }
    return nums;
  }


    /****面试题22：链表中倒数第k个节点
     * 思路1：找出正序中的数值
     * 思路2：快慢指针问题，快指针先走k步，然后慢指针开始走，快指针走到链表末尾时，慢指针指向的位置就是答案
     * 考点：快慢指针呵额鲁棒性
     * ***/
    public ListNode getKthFromEnd(ListNode head, int k){
      //判断k的取值是否合适
      int len = 0;
      ListNode cur = head;
      while(cur != null){
        len +=1;
        cur = cur.next;
      }
      if(k > len || k <= 0){
        return null;
      }
      int kth = len - k;
      for(int i = 1; i<= kth; i++){
        head = head.next;
      }

      return head;

    }
  public ListNode getKthFromEnd1(ListNode head, int k){
      if(head == null || k==0){
        return null;
      }
      ListNode fast = head;
      ListNode slow = head;
      int index = 0;
      while(k-- >0){  //判断k值的取值，提高代码的鲁棒性
        if(fast.next != null){
          fast = fast.next;
        } else {
          return null;
        }
        //index++;
      }
      while(fast != null){
        fast = fast.next;
        slow = slow.next;
      }
      return slow;
  }

  /****链表中环的入口节点
   * x+2a+b = 2(x+a)  x=b
   * ****/
  public ListNode EntryNodeOfLoop(ListNode head){
    if(head == null && head.next == null){
      return null;
    }
    ListNode slow = head;
    ListNode fast = head;

    while(true){
      if(fast == null|| fast.next == null){ //没有环的情况下
        return null;
      }
      slow = slow.next;
      fast = fast.next.next;
      if(fast == slow)break;
    }

    ListNode ptr = head;
    while(ptr != slow){
      ptr = ptr.next;
      slow = slow.next;
    }
    return ptr;
  }

  /***24 反转链表
   * 迭代法：三个指针,分别标记当前节点的，当前节点的上一节点，当前节点的下一节点
   * 递归：
   * ****/
  public ListNode reverseList(ListNode head){
    if(head == null || head.next == null){
      return head;
    }
    ListNode pre = null;
    ListNode cur = head;
    while (cur != null){
      ListNode temp = cur.next;
      cur.next = pre;
      pre = cur;
      cur = temp;
    }
    return pre;
  }

  public ListNode reverseList1(ListNode head){
    if(head == null || head.next == null){
      return head;
    }
    return recur(head, null);

  }
  private ListNode recur(ListNode cur , ListNode pre){
    if(cur == null) return pre;
    ListNode res = recur(cur.next, cur);
    cur.next = pre; //回溯时修改链表指向
    return res;
  }


  /****面试题25 合并两个排序的链表
   * ******/
  public ListNode mergeTwoLists(ListNode l1, ListNode l2){
    if(l1 == null) return l2;
    if(l2 == null) return l1;

    ListNode res = null;
    if(l1.val >= l2.val){
       res = l2;
      res.next = mergeTwoLists(l1, l2.next);
    } else {
      res = l1;
      res.next = mergeTwoLists(l1.next, l2);
    }
    return res;

  }

  /****面试题26 树的子结构
   * *****/
  public boolean isSubStructure(TreeNode A,  TreeNode B){
    boolean res = false;
    if(A != null && B != null){ // 递归终止的条件
      if(A.val == B.val){
        res =  help(A, B);
      }

      if(!res){
        res = isSubStructure(A.left, B) || isSubStructure(A.right, B);
      }

    }
    return res;

  }
  private boolean help(TreeNode A, TreeNode B){
    if(B == null){
      return true;
    }

    if(A == null){
      return false;
    }
    if(A.val != B.val){
      return false;
    }
    return help(A.left , B.left) && help(A.right, B.right);

  }

  /*****27 二叉树的镜像
   * 交换非叶子节点之外的左右树
   * *****/
  public TreeNode mirroTree(TreeNode root){
    if(root == null ){
      return null;
    }
    TreeNode left = mirroTree(root.left);
    TreeNode right = mirroTree(root.right);
    root.left = right;
    root.right = left;
    return root;
  }

  /***28 对称的二叉树
   * 与前序遍历 相对，定义一个对称前序编序，根-右-左
   * 对称儿二叉树的两种遍历结果是一样的
   * ****/
  public boolean isSymmetric(TreeNode root) {
    if(root == null){
      return true;
    }
    return check(root.left, root.right);

  }
  private boolean check(TreeNode l, TreeNode r){
    if(l == r ){
      return true;
    }
    if(l == null || r == null){
      return false;
    }

    if(l.val == r.val && check(l.left, r.right) && check(l.right, r.left)){
      return true;
    } else {
      return false;
    }
  }

  /***29 顺时针打印矩阵
   * ***/
   public int[] printMatrix(int[][] matrix){
     ArrayList<Integer> res = new ArrayList<>();
     if(matrix == null || matrix.length == 0){
       return new int[0];
     }
     int r1 = 0;
     int r2 = matrix.length - 1;
     int c1 = 0;
     int c2 = matrix[0].length - 1;
     while(c1 <= c2 &&  r1 <= r2){
       for(int i = c1; i <= c2; i++){
         res.add(matrix[r1][i]);
       }
       for(int i = r1; i<= r2; i++){
         res.add(matrix[i][c2]);
       }
       if(r1 != r2){ //fixme 防止重复添加
         for(int i = c2-1; i>=c1; i--){ //边界条件
           res.add(matrix[r2][i]);
         }
       }
       if(c1 != c2){ //fixme 防止重复添加
         for(int i = r2-1; i>r1; i--){
           res.add(matrix[i][c1]);
         }
       }

       r1++;
       r2--;
       c1++;
       c2--;
     }
     int[] arr = new int[res.size()];

     return res.stream().mapToInt(Integer::valueOf).toArray(); // 流操作满足转化

   }

   /***30 包含min函数的栈
    * Java 代码中，由于 Stack 中存储的是 int 的包装类 Integer ，因此需要使用 equals() 代替 == 来比较值是否相等。
    * ****/
   class minStack{
     Stack<Integer> data;
     Stack<Integer> min;

     public minStack() {
       data = new Stack<>();
       min = new Stack<>();
     }
     public void push(int x){
       data.push(x);
       if(min.isEmpty()){
         min.push(x);
       } else {
         int minTop = min.peek();
         if(x<minTop){
           min.push(x);
         } else{
           min.push(minTop);
         }
       }
     }
     public void pop(){
        min.pop();
        data.pop();

     }
     public int top(){
       return data.peek();

     }
     public int min(){
       return min.peek();

     }
   }

   /****31 栈的压入 弹出序列
    * 按照入栈序列进行push，每次push时判断栈顶元素是否和出栈序列的元素相等，再进行出栈和出栈序列的循环
    * *****/
   public boolean validateStackSequences(int[] pushed, int[] popped) {
     if(popped.length == 0 && pushed.length == 0){
       return true;
     }
     if(popped == null || popped.length == 0||pushed == null || pushed.length == 0 || popped.length != pushed.length){
       return false;
     }
     Stack<Integer> stack = new Stack<>();
     int len = pushed.length;
     for(int pushIndex  = 0, popIndex= 0; pushIndex<len; pushIndex++){
       stack.push(pushed[pushIndex]);
       while(popIndex < len && !stack.isEmpty() && stack.peek() == popped[popIndex]){
         stack.pop();
         popIndex++;
       }
     }
     return stack.isEmpty();

   }

   /***32 从上到下打印二叉树
    * 思路： 利用双向队列和单向队列，时间较慢
    * 本质是：广度优先遍历
    * *****/
   public int[] levelOrder(TreeNode root) {
     Queue<TreeNode> deque = new LinkedList<>();
     ArrayList<Integer> res = new ArrayList<>();
     if(root != null){ //特殊输入
       deque.add(root);
     }
     while (deque.size() != 0){
       TreeNode node = deque.poll();
       res.add(node.val);
       deque.poll();
       if(node.left != null){
         deque.add(node.left);
       }
       if(node.right != null){
         deque.add(node.right);
       }

     }
     return res.stream().mapToInt(Integer::valueOf).toArray();

   }

  /***32-1 分行从上到下打印二叉树
   * *****/
  public List<List<Integer>> levelOrder1(TreeNode root) {
    if(root == null){
      return  new ArrayList<>();
    }
    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> deque = new LinkedList<>();
    if(root != null){ //特殊输入
      deque.add(root);
    }
    while (!deque.isEmpty()){
      List<Integer> level = new ArrayList<>();
      int n = deque.size(); // 关键点，记录每一层的节点数

      for(int i = 0; i<n; i++){
        TreeNode node = deque.poll();
        level.add(node.val);
        if(node.left != null){
          deque.add(node.left);
        }
        if(node.right != null){
          deque.add(node.right);
        }

      }
      res.add(level);
    }
    return res;
  }

  /***32-2 分行之字从上到下打印二叉树
   * 直接在上述的基础上进行 反转
   * 双端序列，奇数行和偶数行不同的操作
   * *****/
  public List<List<Integer>> levelOrder2(TreeNode root) {
    if(root == null){
      return  new ArrayList<>();
    }
    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> deque = new ArrayDeque<>();
    if(root != null){ //特殊输入
      deque.add(root);
    }
    while (!deque.isEmpty()){
      List<Integer> level = new ArrayList<>();
      int n = deque.size(); // 关键点，记录每一层的节点数

      for(int i = 0; i<n; i++){
        TreeNode node = deque.poll();
        level.add(node.val);
        if(node.left != null){
          deque.add(node.left);
        }
        if(node.right != null){
          deque.add(node.right);
        }

      }
      res.add(level);
    }
    //在上述结果上，反转偶数行的队列
    for(int i=1;i<res.size();i+=2){
      Collections.reverse(res.get(i));
    }
    return res;
  }
  public List<List<Integer>> levelOrder22(TreeNode root) {
    if(root == null){
      return  new ArrayList<>();
    }
    List<List<Integer>> res = new ArrayList<>();
    Deque<TreeNode> deque = new ArrayDeque<>();
    if(root != null){ //特殊输入
      deque.add(root);
    }
    while (!deque.isEmpty()){
      List<Integer> level = new ArrayList<>();
      int n = deque.size(); // 关键点，记录每一层的节点数
      //奇数层，从左往右
      for(int i = 0; i<n; i++){
        TreeNode node = deque.removeFirst();
        level.add(node.val);
        if(node.left != null){
          deque.addLast(node.left);
        }
        if(node.right != null){
          deque.addLast(node.right);
        }

      }
      res.add(level);

      if(deque.isEmpty()) break; // 若为空则提前跳出
      //偶数层 从右往左
      int n1 = deque.size();
      level = new ArrayList<>();
      for(int i = 0; i<n1; i++){
        TreeNode node = deque.removeLast();
        level.add(node.val);
        if(node.right != null){
          deque.addFirst(node.right);
        }
        if(node.left != null){
          deque.addFirst(node.left);
        }

      }
      res.add(level);

    }

    return res;
  }

  /***33 二叉搜索数的后序遍历序列
   * 递归，注意数组的开始索引和结束索引
   * *****/
  public boolean verifyPostorder(int[] postorder) {
    if(postorder == null || postorder.length == 0){
      return false;
    }
    return recur(postorder, 0, postorder.length -1);

  }

  private boolean recur(int[] postorder, int start, int end){
    if(start >= end){
      return true;
    }
    int root = postorder[end];
    int rightIndex = start;
//    for(int i = start; i<= end; i++){
//      if(postorder[i] > root){
//        rightIndex = i;
//        break;
//      }
//
//    }
    while(rightIndex < end && postorder[rightIndex] < root){
      rightIndex++;
    }
    for(int j = rightIndex; j<= end; j++){
      if(postorder[j] < root){
        return false;
      }
    }
    boolean left = recur(postorder, start,rightIndex - 1 );
    boolean right = recur(postorder, rightIndex, end - 1);
    return left&&right;

  }

  /****34 二叉树中和为某一值的路径-------------------------------important
   * 二叉树搜素问题：回溯
   * 从根节点开始的遍历，先序遍历，路径中的节点
   * *****/
  List<List<Integer>> res = new ArrayList<>();
  List<Integer> path = new ArrayList<>();
  public List<List<Integer>> pathSum(TreeNode root, int sum) {
    dfs34(root, sum);
    return res;

  }
  private void dfs34(TreeNode root, int sum){
    if(root == null) return;
    path.add(root.val);
    sum -= root.val;
    if(sum == 0 && root.left == null && root.right == null){
      res.add(new ArrayList<>(path)); //创建新对象，这样有效避免对象的更改
    }
    dfs34(root.left, sum);
    dfs34(root.right, sum);
    path.remove(path.size() -1); //进行回溯

  }

  /****35 复杂链表的复制
   * *****/
  class Node{
    int val;
    Node next;
    Node random;

    Node(int val){
      this.val = val;
      this.next = null;
      this.random = null;
    }
  }
  public Node copyRandomList(Node head){
    if(head == null) return null;
    //第一步 将复制后的节点放到原节点的后边
    Node cur = head;
    while(cur != null){
      Node copyNode = new Node(cur.val);
      copyNode.next = cur.next;
      cur.next = copyNode;

      cur = copyNode.next;
    }

    //第二步 将随机节点链接起来
    cur = head;
    while(cur != null){
      Node random = cur.random;
      if(random != null){
        cur.next.random = random.next;
      }
      cur = cur.next.next;
    }

    //第三步 进行拆分  链表进行拆分-----------------------------important
     cur = head;
    Node copyHead = cur.next;
    Node curCopy = cur.next;
    while(cur != null){
      cur.next = cur.next.next;
      cur = cur.next;
      if(curCopy.next != null){
        curCopy.next = curCopy.next.next;
        curCopy = curCopy.next;

      }

    }
    return copyHead;
  }

  /****36 二叉搜索树与双向链表 --------------------------------
   * 难点在于递归
   * ******/
  class Node1 {
    public int val;
    public Node1 left;
    public Node1 right;

    public Node1() {}

    public Node1(int _val) {
      val = _val;
    }

    public Node1(int _val,Node1 _left,Node1 _right) {
      val = _val;
      left = _left;
      right = _right;
    }
  };
  Node1 head = null;
  Node1 pre = null;  //pre 相当于当前排序链表中的最后一个元素
     public Node1 treeToDoublyList(Node1 root) {
      if(root == null){
        return null;
      }
      dfs36(root);
      head.left = pre;   //循环链表哦、】
      pre.right = head;  //进行头节点和尾节点的相互指向，这两句的顺序也是可以颠倒的
      return head;
    }
    private void dfs36(Node1 cur){
      if(cur == null){
        return;
      }
      dfs36(cur.left);
      //pre用于记录双向链表中位于cur左侧的节点，即上一次迭代中的cur,当pre==null时，cur左侧没有节点,即此时cur为双向链表中的头节点
      if(pre != null){
        pre.right = cur;
      } else {
        head = cur;
      }
      cur.left = pre; //pre是否为null对这句没有影响,且这句放在上面两句if else之前也是可以的。
      pre = cur; //pre指向当前的cur
      dfs36(cur.right); //全部迭代完成后，pre指向双向链表中的尾节点

  }

  /****37 序列化二叉树
   * 按照层序的进行序列化
   * *****/
  public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
      if(root == null) return "[]";
      StringBuilder res = new StringBuilder("[");
      Queue<TreeNode> queue = new LinkedList<>();
      queue.add(root);
      while(!queue.isEmpty()){
        TreeNode node = queue.poll();
        if(node != null){
          res.append(node.val+",");
          queue.add(node.left);
          queue.add(node.right);
        } else  {
          res.append("null" + ",");
        }
      }
      res.deleteCharAt(res.length() - 1);
      res.append("]");
      return res.toString();


    }

    // Decodes your encoded data to tree.
//    public TreeNode deserialize(String data) {
//
//    }
  }
  //按照前序进行序列化
//  public class Codec1 {
//
//    // Encodes a tree to a single string.
//    public String serialize(TreeNode root) {
//
//    }
//
//    // Decodes your encoded data to tree.
//    public TreeNode deserialize(String data) {
//
//    }
//  }

  /**** 38. 字符串的排列
   * todo
   * 回溯全排列
   * 不是很理解交换的思想
   * ****/
  List<String> res38 = new LinkedList<>();

  char[] chars ;
  public String[] permutation(String s) {
    chars = s.toCharArray();
    new ArrayList<String>().toArray(new String[0]);
    dfs38(0);
    return res38.toArray(new String[res38.size()]);
  }
  private void dfs38(int x){
    if(x == chars.length - 1){
      res38.add(String.valueOf(chars));
      return;
    }
    HashSet<Character> hashSet = new HashSet<>();
    for(int i = x; i<chars.length; i++){
      if(hashSet.contains(chars[i])) continue;
      hashSet.add(chars[i]);
      swap(i, x);//将 char[i] 固定在第 x 位
      dfs38(x+1);
      swap(i,x);
    }
  }
  private void swap(int a, int b){
    char temp = chars[a];
    chars[a] = chars[b];
    chars[b] = temp;
  }

  /*** 39. 数组中出现次数超过一半的数字
   * 排序后的中点
   * 摩尔投票：是众数的票数加一，不是众数的票数减1，如果众数，则最后票数和一定大于0
   * 如果前i个票数和为0，后边其余数组中的众数不变
   * ****/
  public int majorityElement(int[] nums) {
//   Arrays.sort(nums);
//   int mid = nums.length/2;
//   return nums[mid];
    //摩尔投票问题, 众数一定存在的
    int x = 0;
    int vote = 0;
    for(int num : nums){
      if(vote == 0){
        x = num;
      }
      vote += num == x?1:-1;
    }
    //验证是否为众数
//    int count = 0;
//    for(int num: nums){
//      if(num == x) count++;
//    }
//    return count > nums.length/2? x:0;
    return x;

  }


  /***40. 最小的k个数
   * TopK问题
   * topk 不用对整个数组进行排序，整个数组进行排序的O(nlogn)
   * ***/
   //快排 时间复杂度O（N）
  public int[] getLeastNumbers(int[] arr, int k) {
    if(arr.length == 0 || k >= arr.length){
      return arr;
    }
    return quickSort(arr, k, 0, arr.length - 1);

  }
  private int[] quickSort(int[] arr, int k, int l, int r){
    int left = l;
    int right = r;
    int pivot = arr[l];
    while(left < right){
      while(left < right && arr[right] >= pivot) right--;
      arr[left] = arr[right];

      while(left < right && arr[left] <= pivot) left++;
      arr[right] = arr[left];
    }
    arr[left] = pivot;
    if(left > k) return quickSort(arr, k, l, left-1); //对左部分再次划分
    if(left < k) return quickSort(arr, k, left+1, r); //对右部分进行划分
    return Arrays.copyOf(arr, k); //刚好就是左边的部分

  }

 //大根堆(前 K 小) / 小根堆（前 K 大)
  //大根堆，每次取出的都是最大的元素
  //O(NLogk)
  public int[] getLeastNumbers1(int[] arr, int k) {
    if(arr.length == 0 || k >= arr.length){
      return arr;
    }
    PriorityQueue<Integer> queue = new PriorityQueue<>((v1, v2)-> v2 -v1);
    for(int a: arr){
      if(queue.size() < k){
        queue.add(a);
      } else if(queue.peek() > a){
        queue.poll();
        queue.add(a);
      }
    }
    int[] res = new int[queue.size()];
    int idx = 0;
    for(int i: queue){
      res[idx++] = i;
    }
    return res;

  }

  /***41. 数据流中的中位数
   * 大顶堆和小顶堆的对撞
   * 小顶堆存放较大部分
   * 大顶堆存放较小部分
   * 奇数的时候，让小顶堆多存放一个
   * ***/
  class MedianFinder {
    PriorityQueue<Integer> big;
    PriorityQueue<Integer> small;

    /** initialize your data structure here. */
    public MedianFinder() {
      small = new PriorityQueue<>();
      big = new PriorityQueue<>((v1, v2) -> v2-v1) ;

    }

    public void addNum(int num) {
      if (small.size() != big.size()){
        small.add(num);
        big.add(small.poll());
      } else {
        big.add(num);
        small.add(big.poll());
      }
    }

    public double findMedian() {
      return big.size()==small.size()?(double)(big.peek() + small.peek())/2:small.peek();

    }
  }

  /** 42. 连续子数组的最大和
   * ****/
  public int maxSubArray(int[] nums) {
    if(nums == null ||nums.length == 0){
      return 0;
    }
    int len = nums.length;
    int[] dp = new int[len+1];
    int res = Integer.MIN_VALUE;
    for(int i = 1; i <= len; i++){
      dp[i] = Math.max(dp[i-1] + nums[i-1], nums[i-1]);
      res = Math.max(dp[i], res);
    }
    return res;
  }

  /**** 44. 数字序列中某一位的数字
   * 注意数字范围，类型要用long
   * *****/
  public int findNthDigit(int n) {
    if(n <0){
      return -1;
    }
    int digit = 1;
    long start = 1;
    long count = 9;
    // 确定digit, 和start，
    while(n - count > 0){
      n -=  count;
      digit++;
      start *= 10;
      count = start * digit * 9;
    }
    long num = start + (n-1)/digit; //确定具体在哪个数字内
    int index = (n-1)%digit;
    return Long.toString(num).charAt(index) - '0';

  }


  /*****45. 把数组排成最小的数
   * 排序规则的传递性证明： x+y>y+x 则x>y
   * 自定义排序的过程
   *  lanmbda
   * *****/
  public String minNumber(int[] nums) {
    String[] strs = new String[nums.length];
    for(int i = 0; i<nums.length; i++){
      strs[i] = String.valueOf(nums[i]);
    }
    Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x)); //自定义排序
    StringBuilder res = new StringBuilder();
    for(String s : strs){
      res.append(s);
    }
    return res.toString();

  }

  /****46. 把数字翻译成字符串
   * ****/
  public int translateNum(int num) {
    String str = String.valueOf(num);
    int len = str.length();
    int[] dp  = new int[len+1];
    dp[0] = 1;
    dp[1] = 1;
    for(int i = 2; i<= len; i++){
      int temp = Integer.valueOf(str.substring(i-2, i));//前闭后开区间
      if(temp<=25 && temp >= 10){
        dp[i] = dp[i-1] + dp[i-2];
      }else {
        dp[i] = dp[i-1];
      }
    }
    return dp[len];

  }

  /****47. 礼物的最大价值
   * 动态规划：dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
   *  *****/
  public int maxValue(int[][] grid) {
    if(grid == null ||grid.length == 0){
      return 0;
    }
    int row = grid.length;
    int col = grid[0].length;
//    int[][] dp = new int[row][col];
//    dp[0][0] = grid[0][0];
//    //注意此时的边界条件，对于首行或者首列，要先进行初始化
//    for(int i = 1; i<row; i++){
//      dp[i][0] = dp[i-1][0] + grid[i][0];
//    }
//    for(int j = 1; j<col; j++){
//      dp[0][j] = dp[0][j-1] + grid[0][j];
//    }
//    for(int i = 1; i< row; i++){
//      for(int j = 1; j<col; j++){
//        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]) + grid[i][j];
//      }
//    }
    //fixme 对数组进行补充一行一列，更加简洁
    int[][] dp = new int[row+1][col+1];
    for(int i = 1; i<=row; i++){
      for(int j = 1; j<=col; j++){
        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]) + grid[i-1][j-1];
      }
    }
    return dp[row][col];
//    return dp[row-1][col-1];

  }

  /****48. 最长不含重复字符的子字符串
   * 动态规划dp[j] 表示j位置结尾的最长不重复字符串的长度
   * dp[j]的长度就取决与j位置左边的字符中与j位置相同的字符的位置
   * *****/
  public int lengthOfLongestSubstring(String s) {
    HashMap<Character, Integer> hashMap = new HashMap<>();
    char[] chars = s.toCharArray();
    int temp  = 0;
    int res = 0;
    for(int i = 0; i<chars.length; i++){
      int l = hashMap.getOrDefault(chars[i], -1);
      hashMap.put(chars[i], i);
      temp = temp >= i - l?i-l:temp+1; //
      res = Math.max(temp, res);
    }
    return res;

  }
  public int lengthOfLongestSubstring1(String s) {
    HashMap<Character, Integer> hashMap = new HashMap<>();
    char[] chars = s.toCharArray();
    int res = 0;
    int l = -1;
    for(int i = 0; i<chars.length; i++){
      if(hashMap.containsKey(chars[i])){
        l = Math.max(l, hashMap.get(chars[i]));//更新左指针
      }
      hashMap.put(chars[i], i); //哈希表记录索引位置
      res = Math.max(res, i -l);
    }
    return res;
  }
  //滑动窗口 left right 维护了一个不含重复字符的窗口
  public int lengthOfLongestSubstring2(String s) {
    HashSet<Character> set = new HashSet<>();
    int res = 0;
    for(int l = 0, r = 0; r < s.length(); r++) {
      char c = s.charAt(r);
      while(set.contains(c)) { //左指针一直右移，导致窗口中不含该元素
        set.remove(s.charAt(l++));
      }
      set.add(c);
      res = Math.max(res, r - l + 1);
    }
    return res;
  }

  /***49. 丑数
   * 丑数是某一个丑数*2 *3 *5 得到的，所以其实是三个有序数组合并为一个无重复元素的有序数组
   * ****/
  public int nthUglyNumber(int n) {
    int a = 0, b = 0, c = 0;// 表示指向三个数组的指针
    int[] dp = new int[n];
    dp[0] = 1;
    for(int i = 1; i<n; i++){
      int n1 = dp[a]*2;
      int n2 = dp[b]*3;
      int n3 = dp[c]*5; // 想象三个数组的内容
      dp[i] = Math.min(Math.min(n1,n2), n3);
      if(dp[i] == n1) a++;
      if(dp[i] == n2) b++;
      if(dp[i] == n3) c++;
    }
    return dp[n-1];

  }


    /***50. 第一个只出现一次的字符
     * 有序哈希表
     * 桶思想: 关于字符和哈希表的题，应该想到桶思想
     * ****/
  public char firstUniqChar(String s) {
    char[] chars = s.toCharArray();
    LinkedHashMap<Character, Integer> hashMap = new LinkedHashMap<>();
    for(char ch: chars){
     int cnt =  hashMap.getOrDefault(ch, 0) + 1;
     hashMap.put(ch, cnt);
    }

    for(Map.Entry<Character, Integer> entry : hashMap.entrySet()){
      if(entry.getValue() == 1){
        return entry.getKey();
      }
    }
    return ' ';

  }
  public char firstUniqChar1(String s) {
    int[] map = new int[26];
    char[] chars = s.toCharArray();
    for(int i = 0; i<s.length(); i++){
      map[chars[i] -'a']++;
    }

    for(int i = 0; i<s.length(); i++){
      if(map[chars[i] - 'a'] == 1){
        return chars[i];
      }
    }
    return ' ';

  }
  /***52. 两个链表的第一个公共节点
   * *****/
  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if(headA == null || headB == null) return null;

    ListNode curA = headA;
    ListNode curB = headB;

   while(curA != curB){
     curA = curA == null?headB:curA.next;
     curB = curB == null?headA:curB.next;
   }
    return curA;
  }

  /*****53 - I. 在排序数组中查找数字 I
   * *****/
  public int search(int[] nums, int target) {
    if(nums == null || nums.length == 0){
      return 0;
    }
    int len = nums.length;
    int left = 0;
    int right = len - 1;
    int res = 0;
    int mid = left + (right - left)/2;
    while(left <= right){ //只有一个元素时，
       mid = left + (right - left)/2;
      if(nums[mid] < target){
        left = mid + 1;
      }else if(nums[mid] > target){
        right = mid - 1;
      } else{
        res += 1;
        break;
      }
    }
    if(nums[mid] == target){
      int k = mid -1;
      while(k >= 0 && nums[k] == target){
        k--;
        res++;
      }
      k = mid+1;
      while(k <= len - 1 && nums[k] == target){
        k++;
        res++;
      }
    }
    return res;


  }

  /**** 54. 二叉搜索树的第k大节点
   * 中序遍历后得到的就是有序
   * *****/
  public int kthLargest(TreeNode root, int k) {
    List<Integer> list = new ArrayList<>();
    dfs54(root, list);
    int len = list.size();
    if(k > len){
      return -1;
    } else {
      return list.get( len - k);
    }

  }
  private void dfs54(TreeNode node, List<Integer> list){
    if(node == null){
      return;
    }
    dfs54(node.left, list);
    list.add(node.val);
    dfs54(node.right, list);
  }

  int count = 0;
  int ans = 0;
  public int kthLargest1(TreeNode root, int k) {
    dfs541(root, k);
    return ans;

  }
  private void dfs541(TreeNode node, int k){
    if(node.right != null){
      dfs541(node.right, k);
    }
    if(count++ == k){
      ans = node.val;
      return;
    }
    if (node.left != null) dfs541(node.left, k);
  }

  /****55 二叉树的深度
   * DFS： 递归
   * BFS: 根据队列
   * ****/
  public int maxDepth(TreeNode root){
    if(root == null){
      return 0;
    }
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
  }
  public int maxDepth1(TreeNode root){
    if(root == null){
      return 0;
    }
    Queue<TreeNode> queue  = new LinkedList<>();
    int res = 0;
    queue.add(root);
    while(!queue.isEmpty()){
      int n = queue.size();
      for(int i = 0; i<n; i++){
        TreeNode node = queue.poll();
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
      }
      res++;

    }
    return res;
  }
  /****55 - II. 平衡二叉树
   * *****/
  public boolean isBalanced(TreeNode root) {
    if(root == null){
      return true;
    }
    int left = maxDepth(root.left);
    int right = maxDepth(root.right);
    if(Math.abs(right - left) >1){
      return false;
    } else {
      return isBalanced(root.left) && isBalanced(root.right); //还需判断左右子树都是平衡二叉树
    }
  }

  /*** 56 - I. 数组中数字出现的次数
   *  关键点是将数组进行分组，两个不同的数字分开，相同的数字都分在一组
   *  找到所以a^b结果中为1的某一位xi，与xi位进行与运算可进行划分
   *  取结果中为1的最低位
   * *****/
  public int[] singleNumbers(int[] nums) {
    int a = 0, b = 0;
    int n = 0;
    for(int num : nums){
      n ^= num;
    }
    int mask = 1; // 表示的位号：两个不同的数在这个位置上不同时为0 或者为1，
    // 找到结果中为1 的最低位
    while((n & mask) == 0){
      mask <<= 1;
    }
    for(int num: nums){
      if((num & mask) != 0){
        a ^= num;
      } else {
        b ^= num;
      }
    }
    return new int[]{a, b};
  }


  /****56 - II. 数组中数字出现的次数 II
   * 适用于一个数组中只有一个数字出现了1次，其余数字都出现了奇数次的情况
   *
   * ****/
  public int singleNumber(int[] nums) {
    if(nums.length < 4){
      return  -1;
    }
    int[] bitNums = new int[32];
    for(int num: nums){
      int mask = 1;
      for(int i = 31; i>=0; i--){
        if((num & mask) != 0){ //里判断条件也可以写为(num&bitMask)==bitMask,而不是==1
          bitNums[i]++;
        }
        mask <<= 1; //左移 左移没有无符号、带符号的区别，都是在右侧补0
      }
    }
    int res = 0;
    for(int i = 0; i<32; i++){
      res <<= 1;
      if(bitNums[i] % 3 == 1){
        res += 1;
      }

    }
    return res;


  }

  /****57 和为s的两个数字
   * 使用hash表，一次遍历，时间O(N)，空间O(N)，
   * 时间O(N) 空间O(1)
   * ****/
  public int[] twoSum(int[] nums, int target) {
    HashSet<Integer> hashSet = new HashSet<>();

    for(int num: nums){
      if(!hashSet.contains(target - num)){
        hashSet.add(num);

      } else {
        return new int[]{num, target - num};
      }
    }
    return new int[]{};

  }

    public int[] twoSum1(int[] nums, int target) {
    if(nums == null || nums.length == 0){
      return new int[2];
    }
    int[] res = new int[2];
    int len = nums.length;
    int i = 0;
    int j = len - 1;
    while (i <j){
      if(nums[i]+nums[j] < target){
        i++;
      } else if(nums[i] + nums[j] > target){
        j--;
      } else{
        res[0] = nums[i];
        res[1] = nums[j];
        return res;
      }
    }

    return new int[2];
  }

  /****Offer 57 - II. 和为s的连续正数序列
   * ****/
  //暴力法
  public int[][] findContinuousSequence(int target) {
    List<List<Integer>> list = new ArrayList<>();
    for(int i = 1; i<= target/2; i++){
      List<Integer> l = new ArrayList<>();
      int tempI = i;
      int tempTarget = target;
      while(tempTarget > 0){
        tempTarget -= tempI;
        l.add(tempI);
        tempI++;
        if(tempTarget == 0){
          list.add(l);
        }
      }
    }

    int[][] res = new int[list.size()][];
    for(int i = 0; i< list.size(); i++){
      res[i] = new int[list.get(i).size()];
      for(int j = 0; j<res[i].length; j++){
        res[i][j] = list.get(i).get(j);
      }
    }
    return res;
  }
  //滑动窗口双指针
  public int[][] findContinuousSequence1(int target) {
    List<int[]> res = new ArrayList<>();
    int i = 1;
    int j = 2;
    int sum = 3;
    while(i < j){
      if(sum == target){
        int[] ans = new int[j-i+1];
        for(int k = i; k<=j; k++){
          ans[k-i] = k;
        }
        res.add(ans);
        sum -= i;
        i++;
      }else if(sum < target){
        j++;
        sum += j;

      } else {
        sum -= i;
        i++;

      }

    }
    return res.toArray(new int[0][]);

  }

  /***58 - I. 翻转单词顺序
   * ****/
  public String reverseWords(String s) {
    String[] str = s.trim().split(" ");
    StringBuilder sb = new StringBuilder();
    int len = str.length;
    for(int i = len-1; i>=0; i--){
      if (str[i].equals("")) {  //去除多个空格
        continue;
      }
      sb.append(' ');
      sb.append(str[i]);
    }
    return sb.toString().trim();

  }
  /****58 - II. 左旋转字符串
   * ****/
  public String reverseLeftWords(String s, int n) {
    String str1 = s.substring(0, n);
    String str2 = s.substring(n);
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append(str2);
    stringBuilder.append(str1);
    return stringBuilder.toString();

  }
  /**** 59 - I. 滑动窗口的最大值
   * 单调双端队列 todo
   * ****/
  public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length == 0 || k<=0){
      return new int[0];
    }
    int len = nums.length;
    int[] res = new int[len - k + 1];
    int max = nums[0];
    for(int i = 0; i<k; i++){
      max = Math.max(max, nums[i]);
    }
    res[0] = max;
    for(int j = 1; j<len - k + 1; j++){
      int pre = nums[j-1];
      if(pre != max){ //如果丢弃的不是上一个窗口的最大值，则需要判断上一个窗口中的最大值和新增加的值的大小
        max = Math.max(max, nums[j+k-1]);
      } else { //刚好丢弃了最大值，那么就要挨个比较区间内的大小
        max = nums[j];
        for(int i = j; i<j+k; i++){
          max = Math.max(nums[i], max);
        }

      }
      res[j] = max;
    }
    return res;
  }

  public int[] maxSlidingWindow1(int[] nums, int k) {
    if (nums == null || nums.length == 0 || k<=0){
      return new int[0];
    }
    int len = nums.length;
    int[] res = new int[len - k + 1];
    Deque<Integer> deque = new LinkedList<>();
    int index = 0;
    for(int i = 0; i<k; i++){
      while(!deque.isEmpty() && nums[i] > deque.peekLast()){
        deque.removeLast();
      }
      deque.addLast(nums[i]);
    }
    res[index++] = deque.peekFirst();
    for(int i = k; i<nums.length; i++){
      if (deque.peekFirst() == nums[i - k]) {
        deque.removeFirst();
      }
      while (!deque.isEmpty() && nums[i] > deque.peekLast()){
        deque.removeLast();
      }
      deque.addLast(nums[i]);

      res[index++] = deque.peekFirst();

    }

    return res;
  }
  /****59 - II. 队列的最大值
   * 辅助队列maxQueue: 维护一个单调递减的队列
   * *****/
  class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> maxQueue;

    public MaxQueue() {
      queue = new LinkedList<>();
      maxQueue = new LinkedList<>();
    }

    public int max_value() {
      if(maxQueue.isEmpty()){
        return -1;
      }else{
        return maxQueue.peekFirst();
      }
    }

    public void push_back(int value) {
      queue.add(value);
      while(!maxQueue.isEmpty() && value > maxQueue.peekLast()){
        maxQueue.removeLast();
      }
      maxQueue.addLast(value);

    }

    public int pop_front() {
      if(queue.isEmpty()){
        return -1;
      }
      if(queue.peek().equals(maxQueue.peekFirst())){ /////注意此处要用equals 代替 == 因为队列中存储的是 int 的包装类 Integer
        maxQueue.removeFirst();
      }
      return queue.poll();

    }
  }

  /*** 60. n个骰子的点数
   * 动态规划和概率知识：dp[n][s] 表示n个骰子掷出点数和为s的概率
   * dp[n][s] += dp[n-1][s-k] * dp[1][k]
   * * ***/
  public double[] dicesProbability(int n) {
    double[] res = new double[5*n+1];
    double[][] dp = new double[n+1][6*n +1] ;
    for(int k = 1; k<=6; k++){
      dp[1][k] = 1.0/6.0;
    }
    for(int i = 2; i<=n; i++){ // n个骰子
      for(int j = i; j<=6*i; j++){ //n个骰子能掷出的点数
        for(int k = 1; k <= 6; k++){
          if(j > k){
            dp[i][j] += dp[i-1][j-k] * dp[1][k];
          }
        }

      }
    }
    for(int i = n; i<= 6*n; i++){
      res[i-n] = dp[n][i];
    }

    return res;


  }

  /****61. 扑克牌中的顺子
   * ****/
  public boolean isStraight(int[] nums) {
    Arrays.sort(nums);
    int zeroCnt = 0;
    int dif = 0;
    for(int i = 0; i<nums.length -1; i++){
      if(nums[i] == 0){
        zeroCnt++;
      } else{
        if(nums[i] == nums[i+1]){
          return false;
        }
        if(nums[i] + 1 != nums[i+1]){
          dif += nums[i+1] - nums[i] - 1;
        }
      }
    }
    return dif <= zeroCnt;


  }

  /***** 62. 圆圈中最后剩下的数字
   * 约瑟夫环
   * *****/
  public int lastRemaining(int n, int m) {
    ArrayList<Integer> list = new ArrayList<>();
    for(int i = 0; i<n; i++){
      list.add(i);
    }
    int idx = 0;
    while(n>1){
     idx = (idx + m - 1)%n; //因为是环，所以取模
     list.remove(idx);
     n--;
    }
    return list.get(0);

  }

  //反推 todo
  public int lastRemaining1(int n, int m) {
    int ans = 0;
    for(int i = 2; i<=n; i++){
      ans = (ans+m)%i;
    }
    return ans;


  }

  /***63. 股票的最大利润
   * dp[i] 表示第i日的最大利润
   * dp[i]=max(dp[i-1], 第i日的价格-前i日内的最小价格)
   * ****/
  public int maxProfit(int[] prices) {
    if(prices == null || prices.length == 0){
      return 0;
    }
//    int len = prices.length;
//    int[] dp = new int[len];
//    int minCost = Integer.MAX_VALUE;
//    int res = 0;
//    dp[0] = 0;
//    for(int i = 1; i<len; i++){
//      minCost = Math.min(minCost, prices[i-1]);
//      dp[i] = Math.max(dp[i-1], prices[i] - minCost);
//      res = Math.max(res, dp[i]);
//    }
//    return res;

    //动态规划的化简
    int cost = Integer.MAX_VALUE;
    int profit = 0;
    for(int price : prices){
      cost  = Math.min(cost, price);
      profit = Math.max(profit, price - cost);
    }
    return profit;

  }

  /****64. 求1+2+…+n
   * 有限制条件，所以不能用等差数列公式
   * 逻辑运算符的短路效应
   * 在 n>1时进行递归
   * n>1不满足时停止递归
   * ***/
  int res64 = 0;
  public int sumNums(int n) {
    boolean flag = n >1  && sumNums(n-1) > 0;
    res64 += n;
    return res64;
  }




  /*****65. 不用加减乘除做加法
   * 位运算：
   * 无进位和：异或规律
   * 进位和：与运算加左右移位
   * *****/
  public int add(int a, int b) {
    while(b != 0){
      int c = (a&b) << 1;
      a = a^b;
      b = c;
    }
    return a;
  }

  /*** 66. 构建乘积数组
   * 本质是两个dp数组，然后左右再相乘
   * **/
  public int[] constructArr(int[] a) {
    int n = a.length;
    if(n == 0){
      return new int[0];
    }
    int[] b = new int[n];
    b[0] = 1;
    int temp = 1;
    for(int i = 1;i < n; i++){
      b[i] = b[i-1] *a[i-1];
    }//从左往右累乘

    for(int j = n-2; j>=0; j--){
      temp *= a[j+1] ;
      b[j] *= temp;
    } // 从右往左累乘
    return b;

  }

  /****67. 把字符串转换成整数
   * ****/
  public int strToInt(String str) {
    char[] chars = str.trim().toCharArray();
    int len = chars.length;
    if(len == 0 ){
      return 0;
    }
    boolean isMinus = false;
    if(chars[0] == '+' || chars[0] == '-' || Character.isDigit(chars[0])){
      if(chars[0] == '+' || chars[0]== '-'){
        if(chars[0] == '-'){
          isMinus = true;
        }
        //删除首位
        chars = Arrays.copyOfRange(chars, 1, len);
      }


      long res = 0;
      int index = 0;
      while(index < chars.length && Character.isDigit(chars[index])){ //注意先后顺序
        res *= 10;
        res += chars[index] - '0';
        if(res > Integer.MAX_VALUE){
          return isMinus? Integer.MIN_VALUE: Integer.MAX_VALUE;
        }
        index ++;
      }
      return  isMinus? -(int)res:(int)res;

    } else{
      return 0;
    }


  }



  /***68 - I. 二叉搜索树的最近公共祖先
   * ****/
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    while(root != null){
      if(root.val > p.val && root.val > q.val){
        root = root.left;
      } else if(root.val < p.val && root.val < q.val){
        root = root.right;
      } else {
        break;
      }
    }
    return root;

  }
  public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q){
    if(root == null){
      return null;
    }
    if(root.val > p.val && root.val > q.val){
      return lowestCommonAncestor(root.left, p, q);
    } else if(root.val < p.val && root.val < q.val){
      return lowestCommonAncestor(root.right, p, q);
    }
    return root;

  }
  /***68 - I. 二叉树的最近公共祖先
   *  如果p q在左右两子树的话，祖先为根节点
   *  如果都在左子树，在左子树上寻找
   * ****/
  public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
    if(root == null){
      return null;
    }
    if(root == p || root == q){
      return root;
    }
    TreeNode left = lowestCommonAncestor2(root.left, p, q);
    TreeNode right = lowestCommonAncestor2(root.right, p, q);
    if(right != null && left != null){
      return root;
    }
    if(right != null){
      return right;
    }
    if(left != null){
      return left;
    }
    return null;

  }
  public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
    if(root == null){
      return root;
    }
    List<TreeNode> path1 = new ArrayList<>();
    List<TreeNode> path2 = new ArrayList<>();
    getPath(path1, root, p);
    getPath(path2, root, q);

    TreeNode res = null;
    int n = Math.min(path1.size(), path2.size());
    for(int i = 0; i<n; i++){
      if(path1.get(i) == path2.get(i)){
        res = path1.get(i);
      }
    }
    return res;
  }
  private void getPath(List<TreeNode> path, TreeNode root, TreeNode node){
    if(root == null){
      return;
    }
    path.add(root);
    if(root ==  node){
      return;
    }
    if(path.get(path.size()-1) != node){
      getPath(path, root.left, node);
    }
    if(path.get(path.size()-1) != node){
      getPath(path, root.right, node);
    }
    if(path.get(path.size()-1) != node){
      path.remove(path.size() -1);
    }


  }









    public static void main(String[] args){
     System.out.println("hello offer");
     Offer offer = new Offer();
     offer.lengthOfLongestSubstring2("pwwkew");

  }
}
