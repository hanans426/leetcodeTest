import com.sun.jmx.snmp.SnmpUnknownModelLcdException;

import java.awt.event.ItemEvent;
import java.util.*;

/**
 * User: gaohan
 * Date: 2021/2/25
 * Time: 10:21
 */
public class ReTest {

  static class ListNode{
    int val;
    ListNode next;

    public ListNode(int v){
      val = v;
    }
  }


  /*** 用两个栈实现队列
   * * ***/
  class ListByStack{
    Stack<Integer> s1; //主要栈
    Stack<Integer> s2; //辅助栈

    public ListByStack(){
      s1 = new Stack<>();
      s2 = new Stack<>();
    }
    public void push(int x){
      s1.push(x);

    }
    public int poll(){
      while(!s1.empty()){
        s2.push(s1.pop());
      }
//      int res = 0;
//      res = s2.pop();
//      while(!s2.empty()){
//        s1.push(s2.pop());
//      }
//      return res;
      if(!s2.isEmpty()){
        return s2.pop();
      } else {

        while(!s1.empty()){
          s2.push(s1.pop());
        }
        if(s2.isEmpty()){
          return -1;
        } else {
          return s2.pop();

        }
      }
    }
  }

  /***剑指 Offer 10- I. 斐波那契数列
   * ****/
  public int fib(int n){
    if(n <= 0){
      return 0;
    }
    if(n <= 2){
      return 1;
    }
    // return fib(n-2) + fib(n-1); 会进行很多重复运算，导致溢出

     int[] dp = new int[n + 1];
     dp[0] = 0;
     dp[1] = 1;
     dp[2] = 1;
     for(int i = 2; i<= n; i++){
       dp[i] = dp[i-2]+ dp[i-1];
       if(dp[i] >= 1000000007 ){
         dp[i] -= 1000000007;
       }
     }
     return dp[n];
  }

  /****剑指 Offer 10- II. 青蛙跳台阶问题
   * 动态规划，dp[i]表示第i个台阶有多少步，
   *  dp[i]=dp[i-1] + dp[i-2];
   * *****/
  public int numWays(int n){
    if(n<=0){
      return 1;
    }
    int[] dp = new int[n+1];
    dp[0] = 1;
    dp[1] = 1;
    for(int i = 2; i<= n; i++){
      dp[i] = dp[i-1]+ dp[i-2];
      if(dp[i] >= 1000000007 ){
        dp[i] -= 1000000007;
      }
    }
    return dp[n];

  }

  /***剑指 Offer 11. 旋转数组的最小数字
   * ****/
  public int minArray(int[] nums){
    if(nums == null || nums.length == 0){
      return -1;
    }
    int left = 0;
    int right = nums.length -1;
    while(left < right){
      int mid = left + (right-left)/2;
      if(nums[mid] > nums[right]) { //说明最小元素一定在右侧
        left = mid + 1;
      } else if(nums[mid] <nums[right]){
        right = mid;
      } else{
        right--; //排除right,去重
      }
    }
    return nums[left];

  }
  /***剑指 Offer 12. 矩阵中的路径
   * ***/
  public boolean exist(char[][] board, String s){
    if(board == null || board.length == 0 || board[0].length == 0 || s.length() == 0){
      return false;
    }
    int row = board.length;
    int col = board.length;
    boolean[][] visited = new boolean[row][col];
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if(dfs12(board, visited, i, j, row, col, s, 0)){
          return true;
        }
      }
    }
    return false;

  }
  int[][] position = {{1,0}, {-1,0}, {0, 1}, {0, -1}};
  private boolean dfs12(char[][] board, boolean[][] visited, int i, int j, int row, int col, String s, int index){
    //递归终止的条件
    if(board[i][j] != s.charAt(index)){
      return false;
    } else if(s.length() - 1 == index){
      return true;
    }
    if(board[i][j] == s.charAt(index)){
      visited[i][j] = true;
      for(int k = 0; k<4; k++){
        int newI = i+position[k][0];
        int newJ = j+position[k][1];
        if(newI<row && newJ<col && newI>=0 && newJ >= 0 && !visited[i][j]){ //判断边界条件
          if(dfs12(board, visited, newI, newJ, row, col, s, index+1)){
            return true;
          }

        }
      }
      visited[i][j] = false; // 说明虽然这个位置匹配上了，但是这个位置下找不到其他可以匹配的了，所以要恢复原样

    }
    return false;
  }

  /***剑指 Offer 13. 机器人的运动范围
   * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，
   * 它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
   * 例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
   * ***/
  public int movingCount(int m, int n, int k){
    if(m == 0 || n == 0){
      return 0;
    }
    boolean[][] vistied = new boolean[m][n];
    return dfs(m, n,  k, 0,0, vistied);

  }
  private int dfs(int m, int n, int k, int i, int j, boolean[][] visited){
    if(!check(m, n, i, j, k) || visited[i][j]){ //递归终止的条件
      return 0;
    }
    int res = 0;
    visited[i][j] = true;
    for(int l = 0; l<4; l++){
      int newI = i + position[l][0];
      int newY = j + position[l][1];
      res += dfs(m, n, k, newI, newY, visited);
    }
   //  visited[i][j] = false;
    return res;


  }
  private boolean check(int m, int n, int x, int y, int k){
    int i = x/10;
    int j = x%10;
    int i1 = y/10;
    int j1 = y%10;
    if(i+j+i1+j1 <= k && x<m && y<n && x>=0 && y>=0){
      return true;
    } else {
      return false;
    }
  }

  /***15. 二进制中1的个数
   * ****/
  public int hammingWeight(int n) {
    if(n == 0){
      return 0;
    }
    int res = 0;
    while(n != 0){
      res += n&1; //n的最低位与1 做与运算，如果是1 ，则加一
      n >>>= 1;
    }
    return res;
  }

  /***剑指 Offer 16. 数值的整数次方
   * *****/
  public double myPow(double x, int n) {
    if(x==0 && n<=0){ //x取负值时且n为非正值，为非法输入
      return 0;
    }
    double res = myPownn(x, Math.abs(n));
    if(n < 0){
      res = 1/res;
    }
    return res;


  }


  private double myPownn(double x, int n){
    if(n == 0){
      return  1.0;
    }
    if(n == 1){
      return x;
    }
    double res = myPownn(x, n/2);
    res *= res;
    if(n%2 == 1){
      res *= x;
    }
    return res;
  }

  /***剑指 Offer 17. 打印从1到最大的n位数
   * ****/
  public void printNumbers(int n) {
    if(n <= 0){
      return;
    }
    int nums =(int) Math.pow(10, n);
    for (int i = 1; i <= nums - 1; i++) {
      System.out.println(i);
    }
  }

  /****剑指 Offer 18. 删除链表的节点 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
   * *****/
  public ListNode deleteNode(ListNode head, int val) {
    if(head == null){
      return null;
    }
    if(head.val == val){
      return head.next;
    }
    ListNode temp = head;
    while(temp != null && temp.next != null){
      if(temp.next.val == val){
        temp.next = temp.next.next;
      }
      temp = temp.next;
    }
    return head;

  }


    /**剑指 Offer 24. 反转链表
     * 保存当前节点的前后两个节点，使后节点指向前节点
     * ***/
  public ListNode reverseList(ListNode head){
    if(head == null ||head.next == null){
      return head;
    }
    ListNode pre = null; //当前节点的前一节点
    ListNode cur = head;
    while(cur != null){
      ListNode next = cur.next; //当前节点的后节点
      cur.next = pre;
      pre = cur;
      cur = next;
    }
    return pre;

  }

  public int solution(int[] A) {
    if(A == null || A.length <= 2){
      return 0;
    }
    int res = 0;
    int len = A.length;
    int[] dp = new int[len];
    dp[0] = 0;
    dp[1] = 0;
    int pre_d = A[1] - A[0];
    int cur_d = 0;
    for(int i = 2; i<len; i++){
      cur_d = A[i] - A[i-1];
      if(cur_d == pre_d){
        dp[i] = dp[i-1] + 1;
      }
      pre_d = cur_d;
      res += dp[i];
    }
    return res;

  }
  public int solution0(String a){
    LinkedList<Integer> stack = new LinkedList<>();
    String[] afterProcess = a.split(" ");
    for(int i = 0; i < afterProcess.length; i++){
      if(afterProcess[i].equals("DUP")){
        if(stack.isEmpty()){
          return -1;
        }
        stack.push(stack.peek());

      }else if(afterProcess[i].equals("POP")){
        if(stack.isEmpty()){
          return -1;
        }
        stack.pop();

      }else if(afterProcess[i].equals("+")){
        int[] number = new int[2];
        int j = 0;
        while(!stack.isEmpty() && j < 2){
          number[j] = stack.pop();
          j++;
        }
        if(j < 2){ return -1; }
        long res = number[0] + number[1];
        if(res > Math.pow(2, 20)-1){ return -1; }
        stack.push(number[0] + number[1]);

      }else if(afterProcess[i].equals("-")){
        int[] number = new int[2];
        int j = 0;
        while(!stack.isEmpty() && j < 2){
          number[j] = stack.pop();
          j++;
        }
        if(j < 2){ return -1; }
        if(number[0] < number[1]){ return -1;}
        stack.push(number[0] - number[1]);

      }else{
        stack.push(Integer.valueOf(afterProcess[i]));
      }
    }
    if(stack.isEmpty()){ return -1; }
    return stack.pop();
  }

  public int solution2(int A, int B){
    if(B < A){
      return 0;
    }
    int start = 0;
    int end = 0;
    for(int i = A; i<=B; i++){
      int num = i * 4 + 1;
      int[] arr = isSquare(num);
      if(arr[0] == 1){
        start = arr[1];
        break;
      }
    }
    if(start == 0){
      return 0;
    }

    for(int i = B; i >= A; i--){
      int num = i * 4 + 1;
      int[] arr = isSquare(num);
      if(arr[0] == 1){
        end = arr[1];
        break;
      }
    }
    return end - start + 1;
  }
  private int[] isSquare(int num){
    int[] res = new int[2];
    if(num<0){
      return res;
    } else {
      for(int i = 0; i<= num/2; i++){
        if(i*i == num){
          res[0] = 1;
          res[1] = (i-1)/2;
          return res;
        }
      }
    }
    return res;
  }

  /****16 最接近的三数之和
   * ****/
  public int threeSumClosest(int[] nums, int target) {
    if(nums == null || nums.length < 3){
      return -1;
    }
    Arrays.sort(nums);
    int ans = nums[0] + nums[1] + nums[2];
    if(target <= ans){
      return ans;
    };
    //内部使用双指针，减少一次循环
    for(int i = 0; i<nums.length - 2; i++){
      int l = i+1;
      int r = nums.length - 1;
      while(l < r){
        int sum = nums[i] + nums[l] + nums[r];
        if(Math.abs(target - sum) < Math.abs(target - ans)){
          ans = sum;
        }
        if(sum < target){
          l++;
        } else if(sum > target){
          r--;
        } else{
          return target;
        }
      }
    }
    return ans;
  }

  /****18. 四数之和
   * *****/
  public List<List<Integer>> fourSum(int[] nums, int target) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    List<Integer> list = new ArrayList<>();
    for(int i = 0; i<nums.length - 3; i++) {
      if(i > 0 && nums[i] == nums[i-1]) continue; //为了去重
      for(int j = i+1; j<nums.length -2; j++){
        if(j > i+1 && nums[j] == nums[j-1]) continue; //为了去重
        int l = j + 1;
        int r = nums.length - 1;
        while(l < r){
          if(nums[i] + nums[j] + nums[l] + nums[r] < target){
            l++;
          } else if(nums[i] + nums[j] + nums[l] + nums[r] > target){
            r--;
          } else {
            list.add(nums[i]);
            list.add(nums[j]);
            list.add(nums[l]);
            list.add(nums[r]);
            res.add(list);
            list.clear();
            while(l < r && nums[l] == nums[l-1]){
              l++;
            }
            while(l < r && nums[r] == nums[r+1]){
              r--;
            }
            l++;
            r--;
          }
        }
      }

    }

    return res;

  }

  /*****26. 删除排序数组中的重复项
   * ****/
  public int removeDuplicates(int[] nums) {
    int len = nums.length;
    int i = 0; //慢指针
    for (int j = 1; j < nums.length ; j++) {
      if(nums[j] != nums[i]){
        i++;
        nums[i] = nums[j];
      }
    }
    return i+1;
  }

  /****54. 螺旋矩阵
   ****/
  public List<Integer> spiralOrder(int[][] matrix) {
    if(matrix == null || matrix.length == 0 || matrix[0].length == 0){
      return null;
    }
    List<Integer> res = new ArrayList<>();
    int r1 = 0, r2 = matrix.length - 1, c1 = 0, c2 = matrix[0].length - 1;
    while(r1 <= r2 && c1 <= c2){
      for(int i = c1; i <= c2; i++){
        res.add(matrix[r1][i]);
      }
      for(int i = r1 + 1; i <= r2; i++){
        res.add(matrix[i][c2]);
      }
      if(r1 != r2){ //为了去重, 会把数据重复添加
        for(int i = c2 - 1; i >= c1; i--){
          res.add(matrix[r2][i]);
        }
      }
     if(c1 != c2){//为了去重
       for(int i = r2 - 1; i > r1; i--){
         res.add(matrix[i][c1]);
       }
     }
      r1++;
      r2--;
      c1++;
      c2--;
    }

    return res;
  }

  /***55. 跳跃游戏
   * 如果一个索引处不可达，那么这个索引右侧的所有位置都不可达，则该位置可达，左侧位置必须全部可达
   */
  public boolean canJump(int[] nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.length; i++) {
      if(i > maxReach){
        return false;
      }
      maxReach = Math.max(maxReach, nums[i] + i);
    }
    return true;
  }

  /***56. 合并区间
   * ***/
  public int[][] merge(int[][] intervals) {
    if(intervals.length < 2){
      return intervals;
    }
    List<int[]> res = new ArrayList<>();
    Arrays.sort(intervals, (v0, v1) -> v0[0]-v1[0]); //按首位递增排序
    int i = 0;
    int n = intervals.length;
    while(i < n){
      int left = intervals[i][0];
      int right = intervals[i][1];
      while(i <n-1 && right >= intervals[i+1][0]){
        right = Math.max(right, intervals[i+1][1]);
        i++;
      }
      res.add(new int[]{left, right});
      i++;
    }
    return res.toArray(new int[res.size()][2]);
  }

  /***57. 插入区间
   * ***/
  public int[][] insert(int[][] intervals, int[] newInterval) {
    List<int[]> res = new ArrayList<>();
    int l = newInterval[0];
    int r = newInterval[1];
    int index = 0;
    int n = intervals.length;
    while(index < n &&  intervals[index][1] < l){
      res.add(intervals[index]);
      index++;
    }
    while (index < n && intervals[index][0] <= r){
      l = Math.min(l, intervals[index][0]);
      r = Math.max(r, intervals[index][1]);
      index++;
    }
    res.add(new int[]{l, r});

    while(index < n){
      res.add(intervals[index]);
      index++;
    }
    return res.toArray(new int[res.size()][2]);

  }



  /***59. 螺旋矩阵 II
   * ***/
  public int[][] generateMatrix(int n) {
    int[][] res = new int[n][n];
    int r1 = 0, r2 = n - 1, c1 = 0, c2 = n - 1;
    int num = 1;
    int target = n * n;
    while(num <= target){
      for(int i = c1; i <= c2; i++){
        res[r1][i] = num;
        num++;
      }
      for(int i = r1 + 1; i <= r2; i++){
        res[i][c2] = num;
        num++;
      }
      for(int i = c2 - 1; i >= c1; i--){
        res[r2][i] = num;
        num++;
      }
      for(int i = r2 -1; i> r1; i--){
        res[i][c1] = num;
        num++;
      }
      r1++;
      r2--;
      c1++;
      c2--;
    }
    return res;
  }

  /****58. 最后一个单词的长度
   * ***/
  public int lengthOfLastWord(String s) {
    String[] arr = s.split(" ");
    if(arr.length  < 2){
      return 0;
    }
    String str = arr[arr.length - 1];
    return str.length();
  }

  /***61. 旋转链表
   * ***/
  public ListNode rotateRight(ListNode head, int k) {
    if(head == null){
      return null;
    }
    int len = 1;
    ListNode temp = head;
    while(temp.next != null){
      temp = temp.next;
      len++;
    }
    int m = k % len;
    ListNode preHead = head;
    for(int i = 0; i< len - m -1; i++){
      preHead = preHead.next;
    }
    ListNode res = preHead.next;
    preHead.next = null;
    temp.next = head;

    return res;
  }

  /****63. 不同路径 II
   * dp[i][j] = dp[i-1][j] + dp[i][j-1]
   * ******/
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    if(obstacleGrid == null || obstacleGrid.length == 0){
      return 0;
    }
    int m = obstacleGrid.length;
    int n = obstacleGrid[0].length;
    int[][] dp = new int[m][n];
    if(obstacleGrid[0][0] == 0){
      dp[0][0] = 1;
    }
    for (int i = 1; i < m; i++) {
      if(obstacleGrid[i][0] == 0){
        dp[i][0] = dp[i-1][0];
      }
    }
    for (int i = 1; i < n; i++) {
      if(obstacleGrid[0][i] == 0){
        dp[0][i] = dp[0][i-1];
      }
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
          if(obstacleGrid[i][j] == 0){
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
          } else {
            dp[i][j] = 0;
          }
      }
    }
    return dp[m-1][n-1];

  }

  /****66. 加一
   * ***/
  public int[] plusOne(int[] digits) {
    int len = digits.length;
    int i = len - 1;
    int carry = 1;
    while( i >= 0 ){
      int temp = digits[i] + carry;
      int  t = temp % 10;
      carry = temp / 10;
      digits[i] = t;
      i--;
    }
    if(carry >= 1){
      int[] res = new int[len + 1];
      res[0] = carry;
      for (int j = 1; j < res.length; j++) {
        res[j] = digits[j-1];
      }
      return res;
    } else {
      return digits;

    }
  }

  /***71. 简化路径
   * ***/
//  public String simplifyPath(String path) {
//
//  }

  /***75
   * ****/
  public void sortColors(int[] nums) {
    if(nums == null||nums.length < 2){
      return;
    }
    int left = 0;
    int right = nums.length - 1;
    int i = 0;
    while(i < right){
      if(nums[i] == 0){
        swap(nums, i, left);
        left++;
        i++;
      } else if(nums[i] == 1){
        i++;
      } else {
        swap(nums, i, right);
        right--;
      }
    }
  }
  private void swap(int[] nums, int i, int j){
    int temp = nums[i];
    nums[j] = temp;
    nums[i] = nums[j];
  }
  private int cnt = 0;

  public int solution(int n, int m, int k){
    if(m < 0 || n < 0 || k < 0 || n*m < k){
      return 0;
    }
    int len = 0;
    dfs(m, n, k, len);
    return cnt;
  }

  private void dfs(int m, int n, int k, int len) {
    if (n - len > k || k > (n - len) * m) {
      return;
    }
    if (len == n && k == 0) {
      cnt = (cnt + 1) % 1000000007;
      return;
    } else if (len == n) {
      return;
    }
    for (int i = 1; i <= m; i++) {
      len++;
      dfs(m, n, k - i, len);
      len--;
    }
  }


  /***491. 递增子序列
   * ****/
  List<List<Integer>> res = new ArrayList<>();
  public List<List<Integer>> findSubsequences(int[] nums) {
    backtrack(nums, new ArrayList<Integer>(), 0);
    return res;
  }


 private void backtrack(int[] nums,List<Integer> path, int start){
    if(start == nums.length){
      return;
    }
   HashSet<Integer> visited = new HashSet<>();
   for(int i = start; i<nums.length; i++){
     if(visited.contains(nums[i])) continue;
     visited.add(nums[i]);
     System.out.println("++++"+visited);
     if(path.size() == 0||path.get(path.size() - 1) <= nums[i]){
       path.add(nums[i]);
       if(path.size()>=2){
         res.add(new ArrayList<>(path));
         System.out.println(new ArrayList<>(path));
       }
       backtrack(nums, path, i+1);
       path.remove(path.size() - 1);
     }

    }

 }

 /***300. 最长递增子序列
  * dp[i]表示以i结尾的最长递增子序列的长度
  * dp[i] = max(dp[j]) + 1;(j < i)
  * ***/
 public int lengthOfLIS(int[] nums) {
   if(nums == null || nums.length == 0){
     return 0;
   }
   int res = 1;
   int[] dp = new int[nums.length];
   dp[0] = 1;
   for (int i = 1; i < nums.length; i++) {
     int max = 0;
     for(int j = 0; j< i; j++){
       if(nums[i] > nums[j]){
        max = Math.max(dp[j], max);
       }
     }
     dp[i] = max + 1;
     res = Math.max(dp[i], res);
   }
   return res;

 }

   /****673. 最长递增子序列的个数
    * ****/
 public int findNumberOfLIS(int[] nums) {
   if(nums == null || nums.length == 0){
     return 0;
   }
   int res = 0;
   int max = 0;
   int[] dp = new int[nums.length];
   int[] cnt = new int[nums.length]; //记录每个索引处最长子序列的个数,LIS的个数如何增加呢
   Arrays.fill(cnt, 1);
   Arrays.fill(dp, 1);
   dp[0] = 1;
   for (int i = 1; i < nums.length; i++) {
     for(int j = 0; j< i; j++){
       if(nums[i] > nums[j]){
         if(dp[j] + 1 > dp[i]){  //第一次找到LIS 时,因为dp[i]初始值为1
            dp[i] = dp[j] + 1;
            cnt[i] = cnt[j];
         } else if(dp[j] + 1 == dp[i]){ //不是第一次找到这个组合，再次找到时
            cnt[i] += cnt[j];
         }
       }
     }
     max = Math.max(dp[i], max);
   }
   for (int i = 0; i < dp.length; i++) {
     if(dp[i] == res){
       res += cnt[i];
     }
   }

   return res;

 }


 /***牛客：字典序最小的最长递增子序列
  * dp 会超时
  * ***/
 public int[] LIS (int[] arr) {
   int[] dp = new int[arr.length];
   Arrays.fill(dp, 1);
   int len = 0;
   for (int i = 1; i < arr.length; i++) {
     for(int j = 0; j<i; j++){
       if(arr[i] > arr[j] && dp[j] + 1>dp[i]){
         dp[i] = dp[j]+1;
       }
     }
     len = Math.max(dp[i], len);
   }

   //反推得到
   int[] res = new int[len];
   for(int t = len, i = arr.length - 1; i>=0; i--){
     if(dp[i] == t){ //从后往前遍历进行添加
       res[t-1] = arr[i];
       t--;
     }
     if(t<=0) break;
   }
   return res;
 }

  public ListNode sortInList (ListNode head) {
    // write code here
    if(head == null || head.next == null){
      return head;
    }
    return mergeSort(head);
  }
  private ListNode mergeSort(ListNode head){
    if(head.next == null){
      return head;
    }
    ListNode slow = head;
    ListNode fast = head.next.next; //
    while(fast != null && fast.next != null){
      slow = slow.next;
      fast = fast.next.next; //如果链表长度为2时，fast指针会越界，
    }
    ListNode right = mergeSort(slow.next);
    slow.next = null;
    ListNode left = mergeSort(head);
    return merge(left, right);
  }

  private ListNode merge(ListNode l1, ListNode l2){
    ListNode pre = new ListNode(-1);
    ListNode cur = pre;
    while(l1 != null && l2 != null){
      if(l1.val < l2.val){
        cur.next = l1;
        l1 = l1.next;
      } else {
        cur.next = l2;
        l2 = l2.next;
      }
      cur = cur.next;
    }
    cur.next = l1 == null?l2:l1;
    return pre.next;
  }

  public String solve (int M, int N) {
    // write code here
    StringBuilder sb = new StringBuilder();
    while(M >= N ){
      sb.append(M%N);
      M = M/N;
    }
    sb.append(M);
    return sb.reverse().toString();

  }
  public int solution(int[] arr, int n){
   if(n ==1){
     return arr[0];
   }
    int[] dp = new int[n+1];
    dp[0] = 0;
    dp[1] = arr[0];
    for (int i = 2; i <= n; i++) {
      int s1 = dp[i-2] + arr[i];
      int s2 = dp[i-1] + (arr[i]+1)/2;
      dp[i] = Math.max(s1,s2);
    }
    return dp[n];
  }


    public static void main(String[] args) {
    ReTest reTest = new ReTest();
    int[] A1 = {1, 3, 5, 7, 9};
    int[] A2 = {-1,1,3,3,3,2,3,2,1,0};
    int[] A3 = {7,7,7,7};
    int[] A4 = {3, 1, -1,-5, -9};
    int[] A5 = {1,2,8,6,4};
    int[][] A6 = {{1, 2, 3, 4},{5,6,7,8}, {9,10,11,12}};


      String s = "11111";
     int a =  Integer.valueOf(s.charAt(0) -'0');
     List<List<Integer>> res = reTest.findSubsequences(A5);
     int[] aa = reTest.LIS(A5);
     ListNode l1 = new ListNode(4);
     ListNode l2 = new ListNode(5);
     ListNode l3 = new ListNode(2);
     ListNode l4 = new ListNode(1);
     ListNode l5 = new ListNode(3);
     l1.next = l2;
     l2.next= l3;
     l3.next = l4;
     l4.next = l5;
     ListNode newHead = reTest.sortInList(l1);
     String sw = reTest.solve(8,2);

     ArrayList<ArrayList<Integer>> ress = new ArrayList<>();
     // System.out.println(a);

//      Scanner sc = new Scanner(System.in);
//       while(sc.hasNext()){
//         int a = sc.nextInt();
//         int b = sc.nextInt();
//       }

//      System.out.println("a=" + (a+1));
//      System.out.println("b=" + b+1);
//      System.out.println("s=" + s + "aa");


  }





}
