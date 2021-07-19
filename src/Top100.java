import javax.print.DocFlavor;
import java.util.*;
import java.util.zip.DataFormatException;

/**
 * User: gaohan
 * Date: 2020/12/21
 * Time: 10:24
 */
public class Top100 {


  /***1. 两数之和
   * 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
   * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
   *
   * ***/
  public int[] twoSum(int[] nums, int target) {
    int[] res = new int[2];
    for(int i = 0; i<nums.length; i++){
      for(int j=i+1; j<nums.length; j++){
        if(nums[i] + nums[j] == target){
          res[0] = i;
          res[1] = j;
          break;
        }
      }
    }
    return res;

  }
  //利用hashmap 存两个满足条件的数据信息，只需要一次循环，并且要及时return，减少时间消耗
  public int[] twoSum1(int[] nums, int target) {
    int[] res = new int[2];
    HashMap<Integer,Integer> hashMap = new HashMap<>();
    for(int i = 0; i<nums.length; i++){
      if(hashMap.containsKey(nums[i])){
        res[0] = hashMap.get(nums[i]);
        res[1] = i;
        return res;

      }
      hashMap.put(target-nums[i], i);
    }
    return res;
  }

  //如果是有序数组，可以使用双指针的方法
  public int[] twoSum2(int[] nums, int target) {
    for(int i = 0, j = nums.length-1; i < j;){
      int sum = nums[i] + nums[j];
      if(sum == target){
        return  new int[]{i, j};
      } else if(sum < target){
        i++;
      } else {
        j--;
      }
    }
    return  null;

  }

    /*****2. 两数相加
     * 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
     * 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
     * 您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     *
     * todo 链表的知识
     * fixme 未通过
     * *****/
  public class ListNode {
      int val;
      ListNode next;
      ListNode() {}
      ListNode(int val) { this.val = val; }
      ListNode(int val, ListNode next) { this.val = val; this.next = next; }
  }

  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode res = new ListNode(0);
    ListNode p = res;
    int temp = 0;
    while (l1 != null || l2 != null){
      int l1Val = l1!=null? l1.val:0;
      int l2Val = l2!=null? l2.val:0;
      int sumVal = l1Val + l2Val + temp;

      temp = sumVal /10;
      sumVal = sumVal %10;

      p.next = new ListNode(sumVal);

      p = p.next;
      if(l1 != null) l1 = l1.next;
      if(l2 != null) l2 = l2.next;
    }
    if(temp == 1) {
      p.next = new ListNode(temp);
    }

    return res.next;
  }

  /***19. 删除链表的倒数第N个节点
   * ****/
  public ListNode removeNthFromEnd(ListNode head, int n) {
    int index = getLength(head) - n;
    ListNode pre = head;
    if(index == 0){
      return head.next;
    }
    for(int i = 0; i < index - 1; i++){
      pre = pre.next; // 循环完了之后，会定位到要删除结点的前一个结点
    }
    pre.next = pre.next.next;
    return head;


  }
  private int getLength(ListNode head){
    int len = 0;
    while(head != null){
      len++;
      head = head.next;
    }

    return len;
  }

  /****23. 合并K个升序链表
   * 分治：不断缩小规模，再不断扩大
   * 根据合并两个有序链表的知识，
   * ****/

  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l1 == null) return l2;
    if(l2 == null) return l1;
    if(l1.val < l2.val){
      l1.next = mergeTwoLists(l1.next, l2);
      return l1;
    } else {
      l2.next = mergeTwoLists(l1, l2.next);
      return l2;
    }

  }

  public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
    ListNode preHead = new ListNode(-1);
    ListNode prev = preHead;
    while(l1 != null && l2 != null){ //注意边界条件
      if(l1.val < l2.val){
        prev.next = l1;
        l1 = l1.next;
      } else {
        prev.next = l2;
        l2 = l2.next;
      }
      prev = prev.next;
    }

    prev.next = l1 == null? l2:l1;
    return preHead.next;


  }

  //时间复杂度O(NK),逐一合并两个链表
  public ListNode mergeKLists(ListNode[] lists) {
    if(lists.length == 0){
      return null;
    }
    ListNode res = null;
    for(ListNode node : lists){
     res =  mergeTwoLists1(res, node);
    }
    return res;
  }

  public ListNode mergeKLists1(ListNode[] lists) {
    if(lists.length == 0 || lists == null){
      return null;
    }
    return merge(lists, 0, lists.length -1);


  }
  private ListNode merge(ListNode[] list, int l, int h){
    if(l == h){
      return list[l];
    }
    int mid = l + (h - l)/2;
    ListNode l1 = merge(list, l, mid);
    ListNode l2 = merge(list, mid+1, h);
    return mergeTwoLists(l1, l2);
  }


  // 可以先不计算链表的长度，利用快慢两个指针来指示要删除的结点
  public ListNode removeNthFromEnd1(ListNode head, int n) {
    ListNode fast = head;
    ListNode slow = head;
    for(int i = 0; i < n; i++){
      fast = fast.next;
    }// fast取 往后顺延n个数的值
    if(fast == null) {
      return head.next;
    }
    while(fast.next != null){
      fast = fast.next;
      slow = slow.next; //这里的slow 是要删除的前一个结点
    }
    slow.next = slow.next.next;
    return head;

  }

  /****3. 无重复字符的最长子串
   * 滑动窗口算法。利用了hashmap优化
   *
   * ******/

  public int lengthOfLongestSubstring(String s) {
    int left = 0;
    int max = 0;
    HashMap<Character, Integer> hashMap = new HashMap<>();
    for(int i = 0; i<s.length(); i++){
      if(hashMap.containsKey(s.charAt(i))){
        left = Math.max(left, hashMap.get(s.charAt(i)) + 1);
      }
      hashMap.put(s.charAt(i), i);
      max = Math.max(max, i- left+1);  // i其实是right，，i -left + 1 其实就是窗口的长度，每次都取最大值
    }
    return max;

  }


  /***4. 寻找两个正序数组的中位数
   *  有序数组的排序是归并排序的一部分
   * ***/
  // 没有限制时间复杂度的情况下，时间复杂度为
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int m = nums1.length;
    int n = nums2.length;
    int[] res = new int[m+n];
    double r = 0;

    for(int i = 0; i<m; i++){
      res[i] = nums1[i];
    }
    int index = m;
    for(int j = 0; j<n; j++){
      res[index] = nums2[j];
      index++;
    }
    Arrays.sort(res);
    if((m+n)%2 == 0){
      r = (double) (res[(m+n)/2] + res[(m+n)/2 - 1]) / 2;
    } else {
      r = res[(m+n)/2];
    }
    return r;
  }

  // 使用归并排序进行重排，但是时间复杂度仍不满足
  public double findMedianSortedArrays1(int[] nums1, int[] nums2) {
    int m = nums1.length;
    int n = nums2.length;
    int[] nums = new int[m+n];
    double res = 0;
    if(m == 0){
      if(n % 2 == 0){
        return (double) (nums2[n/2] + nums2[n/2 - 1]) / 2;
      } else {
        return (double) nums2[n/2];
      }
    }
    if(n == 0){
      if(m % 2 == 0){
        return (double) (nums1[ m/2] + nums1[m/2 - 1]) / 2;
      } else {
        return (double) nums1[m/2];
      }
    }

    int count = 0;
    int i = 0, j=0;

    //归并排序的过程,时间复杂度是 O(m+n)，空间复杂度是 O(m+n)
    while (count != m+n){

      if(i == m){
        while( j != n){
          nums[count] = nums2[j];
          count++;
          j++;
        }
        break;
      }

      if(j == n){
        while (i != m){
          nums[count] = nums1[i];
          count++;
          i++;
        }
        break;

      }

      if(nums1[i] > nums2[j]){
        nums[count] = nums2[j];
        count++;
        j++;
      } else {
        nums[count] = nums1[i];
        count++;
        i++;
      }

    }

    if(count % 2 == 0){
      return (double) (nums[ count/2] + nums[count/2 - 1]) / 2;
    } else {
      return (double) nums[count/2];
    }
  }

 // todo 要求时间复杂度为log(m+n),就要考虑二分法
//  public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
//
//
//  }


  /****5. 最长回文子串
   * 暴力解法：先列举出所有子串，然后判断是否是回文字符串
   * 动态规划
   * ****/
  public String longestPalindrome(String s) {
    int left = 0;
    int max = 1;
    if(s.length() < 2){
      return s;
    }
    char[] charArray = s.toCharArray();

    for(int i = 0; i<s.length() - 1; i++){
      for(int j = i+1; j< s.length(); j++){
        if( j - i +1 > max && valide(charArray, i ,j)){
          max = j-i+1;
          left = i;
        }
      }

    }
    return s.substring(left, left+max);


  }

  private boolean valide(char[] charArray, int left, int right){
    while (left < right){
      if(charArray[left] != charArray[right]){
        return false;
      }
      left++;
      right--;
    }
    return true;
  }

  //todo 动态规划
  public String longestPalindrome1(String s) {
    String res = "";
    return res;

  }

  /*****11. 盛最多水的容器
   * 左右指针都是向内移动，如果移动长板，那么面积一定减小，因为下一个高度只能小于或者等于短板，而移动短板的话，面积可能后变大
   * ******/
  public int maxArea(int[] height) {
    int len = height.length;
    int max = 0;
    for(int i = 0, j = len - 1; i < j;){
      int h = height[i]>height[j]? height[j]:height[i];
      int w = j - i;
      max = Math.max(max, w*h);
      if(height[i] > height[j]){
        j--;  // 值小的指针向内移动，
      } else {
        i++;
      }

    }
    return max;

  }

//  /****10. 正则表达式匹配
//   * 采用动态规划 todo
//   * ****/
//  public boolean isMatch(String s, String p) {
//
//  }


  /*****15. 三数之和
   * *****/
  public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    if(nums.length < 3){
      return res;
    }
    for(int i = 0; i< nums.length; i++){
      int sum = -nums[i];
      int left = i + 1;
      int right = nums.length - 1;

      if(i>0 && nums[i] == nums[i-1]){
        continue;  // 去重操作，如果该值与上一个值相等，那么就跳过这个值
      }
      while(left < right){
        if(nums[left] + nums[right] == sum){
          List<Integer> list = new ArrayList<>();
          list.add(nums[i]);
          list.add(nums[left]);
          list.add(nums[right]);
          res.add(list);
          while(left < right && nums[left] == nums[left + 1]) left++; //去重，如果两个值相同，则跳过
          while(left < right && nums[right] == nums[right - 1]) right--;
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

  /****17. 电话号码的字母组合
   * 经典回溯问题
   * 1. 常规思路：利用res1 作为中间缓存，res 为空时，将字符加入res1,再加入res;res 不为空时，对res 中的每一个已有元素添加字符
   * 2. 回溯 todo
   * ****/
  public List<String> letterCombinations(String digits) {
    List<String> res = new ArrayList<>();
    List<String> res1 = new ArrayList<>();
    HashMap<Character, List<Character>> map = new HashMap<>();
    map.put('2', new ArrayList<Character>(){{add('a'); add('b'); add('c');}});
    map.put('3', new ArrayList<Character>(){{add('d'); add('e'); add('f');}});
    map.put('4', new ArrayList<Character>(){{add('g'); add('h'); add('i');}});
    map.put('5', new ArrayList<Character>(){{add('j'); add('k'); add('l');}});
    map.put('6', new ArrayList<Character>(){{add('m'); add('n'); add('o');}});
    map.put('7', new ArrayList<Character>(){{add('p'); add('q'); add('r');add('s');}});
    map.put('8', new ArrayList<Character>(){{add('t'); add('u'); add('v');}});
    map.put('9', new ArrayList<Character>(){{add('w'); add('x'); add('y');add('z');}});
    for(char c: digits.toCharArray()){
      if(res.isEmpty()){
        for(char c1:map.get(c)){
          res1.add(String.valueOf(c1));
        }
      } else {
        for(char c1: map.get(c)){
          for(int j = 0; j < res.size(); j++){
            res1.add(res.get(j) + c1);
          }
        }
      }
      res.clear();
      res.addAll(res1);
      res1.clear();

    }
    return res;
  }


  /*****20. 有效的括号
   * 1. 字符串替换的方法
   * 2. 栈的思路：后入的左括号要先闭合，采取栈的数据结构
   * ****/
  public boolean isValid(String s) {
    int len = s.length() / 2;
    for(int i = 0; i < len; i++){
      s = s.replace("()","").replace("{}","").replace("[]", "");
    }
    return s.length() == 0;

  }
  public boolean isValid2 (String s) {
    // write code here
    Stack<Character> stack = new Stack<>();
    for(int i = 0; i<s.length(); i++){
      char c = s.charAt(i);
      if(!stack.empty()){
        char b = stack.peek();
        System.out.println(b);
      }

      if(!stack.empty() && s.charAt(i) == stack.peek()){
        stack.pop();
      } else {
        stack.push(s.charAt(i));
      }
    }
    return stack.empty();
  }
  public boolean isValid1(String s) {
    Stack<Character> stack = new Stack<>();
    for(char c : s.toCharArray()){
      if(c =='(') stack.push(')');
      else if(c == '{') stack.push('}');
      else if(c == '[') stack.push(']');
      else if(stack.empty() || c != stack.pop()) return false;
    }
    return stack.empty();
  }

  /*****22. 括号生成
   * 回溯算法：如果发现不穷举一下就没法知道答案的话，可以考虑回溯
   *
   * *****/
  List<String> res = new ArrayList<>();
  public List<String> generateParenthesis(int n) {
    dfs(n, n,"");
    return res;

  }

  private void dfs(int left, int right, String curStr) {
    if(left == 0 && right == 0){
      res.add(curStr);
      return;
    }
    if(left > 0){ //左括号有剩余的话，
      dfs(left - 1, right, curStr+ "(");
    }

    if(right > left){ //右括号有剩余的，补充右括号
       dfs(left, right-1, curStr + ")");
    }

  }

  /****31. 下一个排列
   * 思路:
   * 1. 先从右往左找第一对升序的两个紧邻的数值i-1 i
   * 2  在索引i- end后找到一个数，是比i-1索引的值大中的的最小值，记其索引为j
   * 3  索引i -1  与索引j处的值交换
   * 4. 对 i - end 的元素进行升序
   * ******/
  public void nextPermutation(int[] nums) {
    int len = nums.length;
    for(int i = nums.length - 1; i>0; i--){
      if(nums[i-1] < nums[i]){
        Arrays.sort(nums, i, len);
        for(int j = i; i<len; j++){
          if(nums[j]>nums[i-1]){
            int temp = nums[i-1];
            nums[i-1] = nums[j];
            nums[j] = temp;
            return;
          }
        }
      }
    }
    Arrays.sort(nums);
    return;

  }

  /*****32. 最长有效括号
   * 动态规划: dp[i] 表示以第i个字符结尾的有效括号的长度
   * 如果是'('，dp[i] = 0;
   * 如果是'）'要根据 i-1 的具体情况来确定
   * i-1 的位置是'（'，dp[i] = dp[i-2] + 2;
   * i-1 的位置是'）'，如果dp[i-1] >0, 则再进行判断
   * ****/
  public int longestValidParentheses(String s) {
    if(s.length() == 0 || s.length() == 1){
      return 0;
    }
    int len = s.length();
    int max = 0;
    int[] dp = new int[len];
    dp[0]= 0;
    if(s.charAt(1) == ')' && s.charAt(0) =='('){
      dp[1] = 2;
    } else {
      dp[1] = 0;
    }
    for(int i = 2; i < len; i++){
      if(s.charAt(i) == '('){
        dp[i] = 0;
      } else {
        if(s.charAt(i-1) == '('){
          dp[i] = dp[i-2] + 2;
        } else {
          if(dp[i - 1] > 0) {
            if( i - dp[i-1] -1 >= 0  && s.charAt(i - dp[i-1] - 1) == '('){
              dp[i] = dp[i-1] + 2;
              if(i - dp[i-1] - 2 >= 0){
                dp[i] = dp[i] + dp[i - dp[i-1] -2];
              }
            } else {
              dp[i] = 0;
            }

          }

        }
      }
      max = Math.max(max, dp[i]);
    }

    return Math.max(max, dp[1]);
  }
  public int longestValidParentheses1(String s) {
    Stack<Integer> stack = new Stack<>();
    int max = 0;
    stack.push(-1);
    for(int i = 0; i<s.length(); i++){
      if(s.charAt(i) == '('){
        stack.push(i);
      } else {
        stack.pop();
        if(stack.empty()){
          stack.push(i);
        } else {
          max = Math.max(max, i - stack.peek());
        }
      }
    }
    return max;
  }

  /******33. 搜索旋转排序数组
   * 数组是部分有序的，所以可以利用二分法来进行查找。
   * *****/
  public int search(int[] nums, int target) {
    int len = nums.length;
    int l = 0;
    int h = len - 1;
    while(l < h) {
      int mid = l + (h - l)/2;

      if(nums[mid] == target) {
        return mid;
      } else if(nums[mid] > nums[h]) { //有序的序列在左侧
        if(nums[l] <= target && nums[mid] > target){
          h = mid - 1;
        } else {
          l = mid + 1;
        }

      } else { //有序的序列在右侧
        if(nums[h] >= target && nums[mid] < target){
          l = mid + 1;
        } else {
          h = mid -1;
        }

      }

    }
    return -1;

  }

  /****46. 全排列
   * 回溯经典算法
   * ******/
  public List<List<Integer>> permute(int[] nums) {
    int len = nums.length;
    List<List<Integer>> res = new ArrayList<>();
    if(len == 0){
      return res;
    }
    List<Integer> path = new ArrayList<>();
    boolean[] used = new boolean[len];
    dfs1(nums, len, 0, path,used,res);
    return res;

  }
  private void dfs1(int[] nums, int len, int depth, List<Integer> path, boolean[] used, List<List<Integer>> res){
    if(depth == len){
      res.add(new ArrayList<>(path)); // 此处添加引用，是对path的一个拷贝
      return; //此时一定要return；
    }

    for(int i = 0; i < len; i++){
      if(used[i]){
        continue; //当前的数已经被使用了，就跳过
      }
      path.add(nums[i]);
      used[i] = true;
      dfs1(nums, len, depth + 1, path, used, res);
      used[i] = false;
      path.remove(path.size() -1); // 最后两行,是回溯操作

    }

  }

  /*****39. 组合总和
   * *****/
  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    int len = candidates.length;
    List<List<Integer>> res = new ArrayList<>();
    if(len == 0){
      return res;
    }


    List<Integer> path = new ArrayList<>();
    dfs2(candidates,len, target,0, path,res);
    return res;


  }
  private void dfs2(int[] candidate, int len, int target,int begin, List<Integer> path, List<List<Integer>> res){
    if(target < 0){
      return; //target 为负数时，不再产生新的孩子结点
    }
    if (target == 0){
      res.add(new ArrayList<>(path));
      return;
    }
    for(int i = begin; i < len; i++){
      path.add(candidate[i]);
      dfs2(candidate, len, target - candidate[i], i, path, res);
      path.remove(path.size() - 1); // 将最后的节点移除，进行回溯
    }

  }

  /*****42. 接雨水
   * 思路1：通过计算每个高度处能够接的雨水，累计得到一共能接的雨水，每个高度处的雨水=左侧最高的高度与右侧最高高度的较小值，减去当前的当高度
   * 动态规划：通过从左，和从右两个遍历，得到当前处左边和右边的高度的最大值，利用两个数组根据上述思路进行求解，减少了时间复杂度
   * 单调栈： todo
   * ******/
  public int trap(int[] height) {
    int res = 0;
    int len = height.length;
    for(int i = 0; i < len; i++){
      int leftMax = 0;
      int rightMax = 0;
      for(int j = i ; j>=0; j--){
        leftMax = Math.max(leftMax, height[j]);
      }

      for(int j = i; j < len; j++){
        rightMax = Math.max(rightMax, height[j]);
      }
      res += Math.min(leftMax, rightMax) - height[i];
    }
    return res;

  }
  public int trap1 (int[] height){
    if (height == null || height.length == 0)
      return 0;
    int res = 0;
    int len = height.length;
    int[] leftMax = new int[len];
    int[] rightMax = new int[len];
    leftMax[0] = height[0];
    rightMax[len - 1] = height[len - 1];
    for(int i = 1; i < len; i++){
      leftMax[i] = Math.max(height[i], leftMax[i - 1]);
    }
    for(int j = len - 2; j >= 0; j--){
      rightMax[j] = Math.max(height[j], rightMax[j+1]);
    }
    for(int i = 0; i<len; i++){
      res += Math.min(leftMax[i], rightMax[i]) - height[i];
    }
    return res;
  }

  /****48. 旋转图像
   * 方法1：通过拷贝数组，没有在原地旋转，增加了空间复杂度
   * 方法2： 自外向内，原地旋转 todo
   * 方法3：两次翻转，一次沿对角线翻转，另一次沿中线翻转，也可实现90度旋转的作用
   * *****/
  public void rotate(int[][] matrix) {
    int r = matrix.length;
    int l = matrix[0].length;
    int[][] res = new int[l][r];
    for(int i = 0; i < r; i++){
      for(int j = 0; j < l; j++){
        res[j][r - i - 1] = matrix[i][j];
      }
    }
    for(int i = 0; i < r; i++){
      for(int j = 0; j < l;j++){
        matrix[i][j] = res[i][j];
      }
    }
  }
  public void rotate2(int[][] matrix){
    if(matrix.length == 0 || matrix.length != matrix[0].length) {
      return;
    }
    // 沿对角线翻转： 相当于旋转270度
    int len = matrix.length;
    for(int i = 0; i<len; i++){
      for(int j = i+1; j<len; j++){
        int temp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = temp;
      }
    }
    //沿中轴线翻转，相当与旋转-180度，两个翻转后相当于旋转了90度
    int mid = len / 2;
    for(int i = 0; i< len; i++){
      for(int j = 0; j < mid; j++){
        int temp = matrix[i][j];
        matrix[i][j] = matrix[i][len - j - 1];
        matrix[i][len - j - 1] = temp;
      }

    }

  }
  /****49. 字母异位词分组
   * 1. 对每个字符串进行排序，字符相同的字符串排序后都相同
   * 2. 字符记数的方法，用一个大小为26的数组，索引处对应的值为相应字母出现的次数，然后将该数组转换为字符串，字符串中可以用#填充0，以此字符串为hashmap 的键值
   * ****/
  public List<List<String>> groupAnagrams(String[] strs) {
    HashMap<String, List<String>> hashMap = new HashMap<>();
    for(String s: strs){
      char[] arrays = s.toCharArray();
      List<String> str = new ArrayList<>();
      Arrays.sort(arrays);
      if(hashMap.containsKey(String.copyValueOf(arrays))){
        hashMap.get(String.copyValueOf(arrays)).add(s);
      } else {
        str.add(s);
        hashMap.put(String.copyValueOf(arrays), str);
      }

    }
    return new ArrayList<List<String>>(hashMap.values());

  }
  public List<List<String>> groupAnagrams1(String[] strs) {
    HashMap<String, List<String>> hashMap = new HashMap<>();
    for(String s : strs){
      int[] count = new int[26];
      for(int i = 0; i< s.length(); i++){
        count[s.charAt(i) - 'a'] ++;
      }
      StringBuffer sb = new StringBuffer();
      for(int j = 0; j < 26; j++){
          sb.append('#');
          sb.append(count[j]);
        }
        String key = sb.toString();
      List<String> list = hashMap.getOrDefault(key, new ArrayList<String>());
      list.add(s);
      hashMap.put(key, list);
      }
      return new ArrayList<List<String>>(hashMap.values());

  }

  /*****55. 跳跃游戏：给定一个非负整数数组，你最初位于数组的第一个位置。
               数组中的每个元素代表你在该位置可以跳跃的最大长度。
               判断你是否能够到达最后一个位置。
   * 题解：如果一个位置可以到达，那么这个位置左侧的所有距离都可到达，所以就要看可达到的最远距离，如果最远距离小于最后一个索引，那么就不可到
   * *****/
  public boolean canJump(int[] nums) {
    int len = nums.length;
    int maxReach = 0;
    for(int i = 0; i < len; i++){
      if(i > maxReach){
        return false;
      }
      maxReach = Math.max(maxReach, nums[i]+i);
    }
    return true;

  }

  /****56. 合并区间:给出一个区间的集合，请合并所有重叠的区间。
   * 先按照首元素进行排序，排序后的重叠的情况只需比较当前值和 返回序列中上一个值的尾值。
   * ******/
  public int[][] merge(int[][] intervals) {
    int len = intervals.length;
    Arrays.sort(intervals, (v0,v1)-> v0[0]-v1[0]);
    int[][] res = new int[len][2];
    int index = 0;
    for(int i = 0; i<len; i++){
      if(i == 0 || intervals[i][0] > res[index - 1][1]){ //第一个元素时，或者当前数组的首值大于返回结果中的后一元素的尾值时，直接在res中加入该数组
        res[index++] = intervals[i];
      } else {
        int right = Math.max(res[index-1][1], intervals[i][1]);
        res[index - 1][1] = right; //更新index -1 处的尾值，此时index 不用更新
      }
    }
    return Arrays.copyOf(res, index);
  }


  /***72. 编辑距离:给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数
   *
   * *****/
//  public int minDistance(String word1, String word2) {
//
//  }

  /***75. 颜色分类: 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
   * 计数排序：一次遍历先计算出每个颜色的个数，然后根据在填充到数组中
   * 荷兰国旗问题：快速排序：设计循环不变量
   * 循环不变量：声明的变量在循环的过程中保持定义不变
   *
   * *****/
  public void sortColors(int[] nums) {
    int zero = 0;
    int one = 0;
    int two = 0;
    int len  = nums.length;
    for(int i = 0; i < len; i++){
      if(nums[i] == 0){
        zero++;
      } else if(nums[i] == 1){
        one++;
      } else {
        two++;
      }
    }

    for(int i = 0; i<zero; i++){
      nums[i] = 0;
    }
    for(int i = zero; i<zero + one; i++){
      nums[i] = 1;
    }
    for(int i = zero + one; i<zero + one + two; i++){
      nums[i] = 2;
    }

    }
  public void sortColors1(int[] nums){
    int len = nums.length;
    if(len < 2){
      return;
    }
    /**
     * all in[0,zero]  = 0，
     * all in(zero, i) = 1
     * all in[two, len - 1] = 2
     * 变量初始化时要保障三个数组都是空
     * */
    int zero = -1; // zero 是区分0 和 1的分界的指针
    int i = 0;  //循环变量
    int two = len; //two 是区分1 和 2 的边界的指针

    while(i < two){ //当i == two 的时候，三个区间刚好完全覆盖整个区间，索引是i==two时，就要终止了，循环条件就是i< two;
      if(nums[i] == 0){
        zero++; //zero 当前处的值也是0，所以要先++，
        swap(nums, i,zero);
        i++;
      } else if(nums[i] == 1){
        i++;
      } else {
        two--; // 开始指向的是len, 所以要先--
        swap(nums, i, two);
        //此时不确定交换回来的是什么，所以不需要i++,还要对这个循环变量再次查询


      }
    }
  }
  private void swap(int[] nums, int i, int j){
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  /*****79. 单词搜索
   * 二维平面上使用回溯 dfs
   * 对二维平面上进行遍历，每个值都有可能作为开始的搜寻的根节点，一但有一个为true,则说明存在，如果遍历了所有的根节点都没有符合的值，那么不存在
   * 确定了根节点后，对匹配word 进行深度优先遍历，其中可变量是word 中字符的索引和当前平面上的坐标
   * 当前坐标的字符与word中相应的字符相等时，对当前坐标进行四个方向上的移动，得到新坐标的值与word中下一个字符进行比较，依次进行
   * 回溯终止的条件是：word 中的字符是最后一个元素时，此时返回最后一个字符是否和二维平面上的字符相同
   * *****/
  public boolean exist(char[][] board, String word) {
    int m = board.length;
    if(m == 0){
      return false;
    }
    int n = board[0].length;
    int[][] position = {{0,-1},{1,0},{0,1}, {-1,0}}; //表示四个移动方位

    boolean[][] visited = new boolean[m][n];

    for(int i = 0; i< m; i++){
      for(int j = 0; j < n; j++){
        if(dfs3(board, m, n, visited, i, j, word,0, position)){
          return true;
        }
      }
    }
    return false;
  }
  //index: 为word中第几个字符
  private boolean dfs3(char[][] board,int row, int col, boolean[][] visited, int i, int j, String word, int index, int[][] position){
    if(index == word.length() - 1){
      return board[i][j] == word.charAt(index); //回溯的终止条件是,对word 的遍历结束，
    }
    if(board[i][j] == word.charAt(index)){
      visited[i][j] = true;
      for(int k = 0; k < 4; k++){
        int newX = i + position[k][0];
        int newY = j + position[k][1];
        if(inArea(row, col, newX, newY) && !visited[newX][newY]){
          if(dfs3(board, row, col, visited, newX, newY, word, index+1, position)){
            return true;
          }

        }
      }
      visited[i][j] = false;
    }
    return  false;



  }

  private boolean inArea(int row, int col, int x, int y){
    if(x < row && y < col && x >=0 && y>=0){
      return true;
    } else {
      return false;
    }

  }


    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


  /****94. 二叉树的中序遍历 给定一个二叉树的根节点 root ，返回它的 中序 遍历。
   * 中序遍历：先左，根节点，最后右
   * 递归：先处理当前节点的左子树，将当前节点的值加入队列，然后再遍历右子树，递归终止的条件是为空节点
   * 用栈做， todo
   * 莫里斯遍历：  todo
   * ****/

  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    inorder(root, res);
    return res;


  }

  private void inorder(TreeNode root, List<Integer> res){
    if(root == null){
      return;
    }
    inorder(root.left, res);
    res.add(root.val);
    inorder(root.right, res);
  }

  /***96. 不同的二叉搜索树
   * 思路：动态规划
   * G(n) 长度为n的序列能构成不同的二叉搜索树的个数
   * F(i,n)以i和为根节点，序列长度为n的不同二叉搜索树的个数
   * F(i,n) = G(i-1) * G(n-i);
   * 笛卡尔积：G（n） =求和[i-n] F(i,n)
   * G[0] = 1;
   * G[1] = 1;
   * ****/
  public int numTrees(int n) {
    int[] dp = new int[n+1];
    dp[0] = 1;
    dp[1] = 1;
    for(int i = 2; i<=n; i++){ //i 表示有多少个序列的长度
      for(int j = 1; j <= i; j++){ //j表示根节点的取值
        dp[i] += dp[j-1]*dp[i-j];
      }

    }
    return dp[n];

  }
  /*****98. 验证二叉搜索树： 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
   * 思路1：递归
   * 思路2：中序遍历：判断是否是递增的
   * ******/
  public boolean isValidBST(TreeNode root) {
    List<Integer> res = inorderTraversal(root); //调用上上题的中序遍历方法，得到升序的队列
    for(int i = 1; i< res.size();i++){
      if(res.get(i) <= res.get(i-1)){
        return false;
      }
    }
    return true;

  }
  long pre = Long.MIN_VALUE;
  public boolean isValidBST1(TreeNode root) {
    if(root == null){
      return true;
    }
    //访问左子树
    if(!isValidBST1(root.left)){
      return false;
    }
    //访问当前节点，如果当前节点小于等于中序遍历的前一个节点，则不满足BST
    if(root.val <= pre){
      return false;
    }
    pre = root.val; //
    //访问右子树
    return isValidBST1(root.right);

  }
  public boolean isValidBST2(TreeNode root) {
    if(root == null){
      return true;
    }
    if(isValidBST2(root.left)){
      if(root.val >= pre){
        pre = root.val;
        return isValidBST2(root.right);
      }
    }
    return false;
  }

  /****101. 对称二叉树 给定一个二叉树，检查它是否是镜像对称的。
   * *****/
  public boolean isSymmetric(TreeNode root) {
    if(root == null){
      return true;
    }else {
      return check(root.left, root.right);
    }


  }
  private boolean check(TreeNode left, TreeNode right){
    if(left == right){
      return true;
    }
    if(left == null || right == null){ // 递归结束的条件
      return false;
    }
    if(left.val == right.val && check(left.left, right.right) && check(left.right, right.left)){ //
      return true;
    } else {
      return false;
    }

  }
  /****102. 二叉树的层序遍历
   * BFS:层序遍历，最短路径
   * todo BFS经典应用
   * ****/
  public List<List<Integer>> levelOrder(TreeNode root) {
    if(root == null){
      return null;
    }
    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> queue = new ArrayDeque<>();
    if(root != null){
      queue.add(root);
    }
    while(!queue.isEmpty()){
      List<Integer> level = new ArrayList<>();
      int n = queue.size();
      for(int i = 0; i<n; i++){
        TreeNode node = queue.poll();
        level.add(node.val);
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }

      }
      res.add(level);
    }
    return res;

  }
  /****104. 二叉树的最大深度
   * 递归
   * BFS
   * *****/
  public int maxDepth(TreeNode root) {
    if(root == null){
      return 0;
    } else {
      return Math.max(maxDepth(root.left), maxDepth(root.right))+1;
    }

  }

  /****105. 从前序与中序遍历序列构造二叉树
   * *****/
  HashMap<Integer, Integer> indexMap = new HashMap<>();
  public TreeNode buildTree(int[] preorder, int[] inorder) {
    int n = preorder.length;
    for(int i = 0; i<n; i++){
      indexMap.put(inorder[i], i);
    }
    return myBuildTree(preorder, inorder, 0, n-1, 0, n-1);

  }

  private TreeNode myBuildTree(int[] preorder, int[] inorder, int preLeft, int preRight, int inLeft, int inRight){
    if(preLeft > preRight){
      return null;
    }
    int preRoot = preLeft;
    int inRoot = indexMap.get(preorder[preRoot]);
    TreeNode root = new TreeNode(preorder[preRoot]);
    int sizeLeft = inRoot - inLeft;
    root.left = myBuildTree(preorder, inorder, preLeft+1, preLeft+sizeLeft, inLeft, inRoot -1);
    root.right  = myBuildTree(preorder, inorder, preLeft+sizeLeft+1, preRight, inRoot+1, inRight);
    return root;

  }

  /***114. 二叉树展开为链表 给定一个二叉树，原地将它展开为一个单链表。
   * 用前序遍历的方法
   * *****/
  public void flatten(TreeNode root) {
    List<TreeNode> list = new ArrayList<>();
    preorder(root,list);
    int n = list.size();
    for(int i = 0; i<n; i++){
      TreeNode pre = list.get(i-1);
      TreeNode cur = list.get(i);
      pre.left = null;
      pre.right = cur;
    }



  }
  private void preorder(TreeNode root,List<TreeNode> list){
    if(root == null){
      return;
    }
    list.add(root);
    preorder(root.left, list);
    preorder(root.right,list);

  }


  /****136. 只出现一次的数
   * 任何数和自己异或都是0，0和任何数异或都是本身，异或满足交换律和结合律，所有数字异得到的是单独的数
   * *****/
  public int singleNumber(int[] nums) {
    int single = 0;
    for(int n : nums){
      single = single^n;
    }
    return single;

  }
  /****139. 单词拆分
   * 动态规划：dp[i]表示字符串s中第i-1结尾的字符串是否满足可被wordDict划分
   * dp[i] = dp[j] && check(s[j, i-1])
   * dp[i] = dp
   * *****/
  public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();
    boolean[] dp = new boolean[n+1];
    dp[0] = true;
    for(int i = 1; i<=n; i++){
      for(int j = 0; j<i; j++){
        if(dp[j] && wordDict.contains(s.substring(j, i))){
          dp[i] = true;
          break;
        }
      }
    }
    return dp[n];
  }

  /***142. 环形链表 II
   * hashSet 方法，将节点加入, 空间复杂度不是O(1)
   * 快慢指针法：s= nb, f = 2nb, s再走a步就到到环形链表入口处
   * ***/
  public ListNode detectCycle(ListNode head) {
    Set<ListNode> set = new HashSet<>();
    ListNode pos = head;
    while (pos != null){
      if(set.contains(pos)){
        return pos;
      } else {
        set.add(pos);
      }
      pos = pos.next;
    }
    return null;
  }
  public ListNode detectCycle1(ListNode head) {
    ListNode fast = head;
    ListNode slow = head;
    ListNode ptr = head;
    while(true){
      if(fast == null||fast.next == null) return null;
      fast = fast.next.next;
      slow = slow.next;
      if(fast == slow) break;
    }
    while(ptr != slow){
      ptr = ptr.next;
      slow = slow.next;
    }
    return ptr;

  }
  /***146. LRU 缓存机制
   * LinkedHashMap:有顺序的hashMap
   * ******/

  class LRUCache1 extends LinkedHashMap<Integer,Integer> {
    private int cap = 0;


    public LRUCache1(int capacity) {
      super(capacity, 0.75F, true);
      this.cap = capacity;
    }

    public int get(int key) {
      return super.getOrDefault(key, -1);

    }

    public void put(int key, int value) {
      super.put(key, value);
    }
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
      return size() > cap;
    }

  }

  class LRUCache {

    class LinkedNode{
      int key;
      int value;
      LinkedNode pre;
      LinkedNode next;
      public LinkedNode(){}
      public LinkedNode(int _key, int _value){key = _key; value = _value;}
    }// 有顺序的

    private HashMap<Integer, LinkedNode> cache = new HashMap<>();
    private int cap = 0;
    private int size = 0;
    private LinkedNode head;
    private LinkedNode tail;

    public LRUCache(int capacity) {
      this.cap = capacity;
      this.size = 0;
      head = new LinkedNode();
      tail = new LinkedNode();
      head.next = tail;
      tail.pre = head;

    }

    public int get(int key) {
      LinkedNode node = cache.get(key);
      if(node == null){
        return  -1;
      }
      //如果node存在，则将该节点移动到头部
      moveToHead(node);
      return node.value;


    }

    public void put(int key, int value) {
      LinkedNode node = cache.get(key);
      if(node == null){
        LinkedNode newNode = new LinkedNode(key, value);
        cache.put(key, newNode);
        //将该node添加到头部
        addToHead(newNode);
        size++;
        if(size > cap){ //如果大于容量，则删除链表的尾部节点
          LinkedNode n = removeTail();
          cache.remove(tail.key);
          size--;
        }
      } else {
        node.value = value;
        //将该节点移动到头部
        moveToHead(node);
      }

    }

    private void addToHead(LinkedNode node){
      node.pre = head;
      node.next = head.next;
      head.next.pre = node;
      head.next = node;

    }
    private void removeNode(LinkedNode node){
      node.pre.next = node.next;
      node.next.pre = node.pre;
    }
    private void moveToHead(LinkedNode node){
      removeNode(node);
      addToHead(node);
    }
    private LinkedNode removeTail(){
      LinkedNode res = tail.pre;
      removeNode(res);
      return res;

    }
  }
  /***148. 排序链表==============
   * 归并排序，
   * 利用快慢指针，找到链表的中点，自顶向下进行划分
   * todo 快排
   * ****/
  public ListNode sortList(ListNode head) {
    return mergeSort(head);

  }
  private ListNode  mergeSort(ListNode head){
    if(head.next == null || head == null){
      return head;
    }
    ListNode slow = head;
    ListNode fast = head.next.next;
    while(fast != null && fast.next != null){
      slow = slow.next;
      fast = fast.next.next;
    }
    ListNode r = mergeSort(slow.next);

    slow.next=null;
    ListNode l = mergeSort(head);
    return merge2List(l, r);


  }
  private ListNode merge2List(ListNode listNode1, ListNode listNode2){
    if(listNode1 == null) return listNode2;
    if(listNode2 == null) return  listNode1;
    if(listNode1.val < listNode2.val){
      listNode1.next = merge2List(listNode1.next, listNode2);
      return listNode1;
    } else {
      listNode2.next = merge2List(listNode1, listNode2.next);
      return listNode2;
    }
  }
//  private ListNode merge2List1(ListNode listNode1, ListNode listNode2){
////    ListNode preHead = new ListNode(-1);
////    ListNode pre = preHead;
////    while(listNode1 != null && listNode2 != null ){
////      if(listNode1.val < listNode2.val){
////        pre.next = listNode1;
////        listNode1 = listNode1.next;
////      } else {
////        pre.next = listNode2;
////        listNode2 = listNode2.next;
////      }
////      pre = pre.next;
////
////    }
////    pre.next =  listNode1 == null?listNode2:listNode1;
////    return preHead.next;
//
//
//  }

  /***152. 乘积最大子数组
   * dp[i]表示以该位结尾的子数组的最大乘积
   *  dp[i] = max(dp[i-1]* nums[i], nums[i])
   *  因为有负数，所以还需要另一个数组最为辅助，记录最小值
   *****/
  public int maxProduct(int[] nums) {
    int n = nums.length;
    if(n == 0){
      return 0;
    }
    int res = nums[0];
    int[] max = new int[n+1];
    int[] min = new int[n+1];
    max[0] = nums[0];
    min[0] = nums[0];
    for(int i = 1; i<n; i++){
      max[i] = Math.max(Math.max(max[i-1] * nums[i], min[i-1]* nums[i]), nums[i]);
      min[i] = Math.min(Math.min(max[i-1] * nums[i], min[i-1]* nums[i]), nums[i]);
      res = Math.max(max[i], res);
    }
    return res;
  }

  /****155. 最小栈
   * 方法1： 用辅助栈
   * 方法2：自定义数据结构弄的
   * ****/
  class MinStack {
    class Node{
      int val;
      int min;
      public Node(int _val, int _min){
        this.val = _val;
        this.min = _min;
      }
    }
    Deque<Node> stack;



    /** initialize your data structure here. */
    public MinStack() {
      stack = new ArrayDeque<Node>();
    }

    public void push(int x) {
      if(stack.isEmpty()){
        stack.push(new Node(x,x));
      } else {
        stack.push(new Node(x, Math.min(x, stack.peek().min)));
      }
    }

    public void pop() {
      stack.pop();

    }

    public int top() {
      return stack.peek().val;

    }

    public int getMin() {
      return stack.peek().min;

    }
  }

  /****
   * 200. 岛屿数量
   *  dfs :深度优先遍历
   */
  public int numIslands(char[][] grid) {
    if(grid == null ||grid.length == 0){
      return 0;
    }
    int r = grid.length;
    int c = grid[0].length;
    int res = 0;
    for(int i = 0; i< r; i++){
      for(int j = 0; j < c;j++){
        if(grid[i][j] == '1'){
          res++;
          dfs4(grid, i, j, r, c);

        }
      }
    }
    return res;

  }
  private void dfs4(char[][] grid, int r, int c, int nr, int nc){
    if(c < 0 || r < 0 || c >= nc || r >= nr || grid[r][c] == '0'){
      return;
    }
    grid[r][c] = '0';
    dfs4(grid, r-1, c, nr, nc);
    dfs4(grid, r+1, c, nr, nc);
    dfs4(grid, r, c+1, nr, nc);
    dfs4(grid, r, c-1, nr, nc);

  }

  /****207. 课程表
   * 拓扑排序Kahn算法
   * todo
   * *****/
//  public boolean canFinish(int numCourses, int[][] prerequisites) {
//
//  }

  /****208. 实现 Trie (前缀树)
   * 非典型的多叉树模型
   * 一次建树，多次查询
   * *****/
  class Trie {

    /** Initialize your data structure here. */

    class TrieNode{
      boolean isEnd;
      TrieNode[] next;

      public TrieNode(){
        isEnd = false;
        next = new TrieNode[26];
      }

    }
    private TrieNode root;
    public Trie() {
      root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
      TrieNode node = root;
      for(char c : word.toCharArray()){
        if(node.next[c -'a'] == null){
          node.next[c-'a'] = new TrieNode();
        }
        node = node.next[c-'a'];
      }
      node.isEnd = true;

    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
      TrieNode node = root;
      for(char c: word.toCharArray()){
        node = node.next[c-'a'];
        if(node == null){
          return false;
        }
      }
      return node.isEnd;

    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
      TrieNode node = root;
      for(char c:prefix.toCharArray()){
        node = node.next[c-'a'];
        if(node == null){
          return false;
        }
      }
      return true;

    }
  }

  /****221. 最大正方形
   * 动态规划：dp(i + 1, j + 1) 是以 matrix(i, j)为右下角的正方形的最大边长
   * dp[i][j]= 1+ min(dp[i-1][j-1],min(dp[i-1][j], dp[i][j-1]))
   * 有边界问题时，可以多增加一行或者一列，预处理为0，
   * ****/
  public int maximalSquare(char[][] matrix) {
    int r = matrix.length;
    int c = matrix[0].length;
    if(r == 0 || c == 0){
      return 0;
    }
    int[][] dp = new int[r+1][c+1]; //多增加一行，一列，相当于已经预处理新增第一行 第一列为0
    int res = Math.max(dp[0][0], Math.max(dp[0][1], dp[1][0]));
    for(int i = 0; i<r; i++){
      for(int j = 0; j<c; j++){
        if(matrix[i][j] == '1'){
          dp[i+1][j+1] = 1 + Math.min(dp[i][j], Math.min(dp[i][j+1], dp[i+1][j]));
        }
        res = Math.max(res, dp[i+1][j+1]);

      }
    }
    return res*res;




  }
  /*****226. 翻转二叉树
   * 递归
   * *****/
  public TreeNode invertTree(TreeNode root) {
    if(root == null){
      return root;
    }

    TreeNode left = invertTree(root.left);
    TreeNode right = invertTree(root.right);
    root.left = right;
    root.right = left;
    return root;

  }

  /***234. 回文链表\
   * 快慢指针，找到链表的中点，划分为两个链表，进行对比是否相同
   *
   * ****/
  public boolean isPalindrome(ListNode head) {
    if(head == null || head.next == null){
      return true;
    }
    ListNode slow = head;
    ListNode fast = head.next.next;
    while(fast != null && fast.next != null){
      slow = slow.next;
      fast = fast.next.next;
    }
    slow = reverseListNode(slow.next);
    while(slow != null){
      if(head.val != slow.val){
        return false;
      }
      slow = slow.next;
      head = head.next;
    }
    return true;

  }
  //链表翻转
  private ListNode reverseListNode(ListNode head){
    if(head.next == null){
      return head;
    }
    ListNode newHead = reverseListNode(head.next);
    head.next.next = head;
    head.next = null;
    return newHead;
  }

  /*****236. 二叉树的最近公共祖先
   * 递归
   * ****/
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if(root == null || root == p || root == q ) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if(left == null && right == null) return null;
    if(left == null) return right;
    if(right == null) return left;
    return root;

  }

  /****240. 搜索二维矩阵 II
   * 两个方向上进行二分，只要一个方向上找到就可
   * 直接进行二维二分
   * *****/
  public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0) return false;
    int m = matrix.length;
    int n = matrix[0].length;
    int row = 0;
    int col = n-1;
    while(row < m && col > 0){
      if(matrix[row][col] > target){
        col--;
      } else if(matrix[row][col] < target){
        row++;
      } else {
        return true;
      }
    }
    return false;

  }

  /***283. 移动零
   * ******/
  public void moveZeroes(int[] nums) {
    int len = nums.length;
    int index = 0;

    for(int i = 0; i<len ; i++){
      if(nums[i] != 0){
        nums[index++] = nums[i];
      }
    }
    while(index < len){
      nums[index++] = 0;
    }

  }

  /***287. 寻找重复数
   * 先排序，再二分
   * 不需要排序，利用一个计数变量进行二分
   * ****/
  public int findDuplicate(int[] nums) {
    if(nums == null ||nums.length == 0){
      return 0;
    }
    int len = nums.length;
    Arrays.sort(nums);
    int res = 0;
    int l = 0;
    int h = len-1;
    while(l < h){
      int mid = l + (h-l)/2;
      if(nums[mid]-1 < mid){
        h = mid;
      } else{
        l = mid +1;
      }
      res = nums[mid];
    }
    return res;

  }
  public int findDuplicate1(int[] nums) {
    int len = nums.length;
    int left = 1;
    int right = len - 1;
    while (left < right) {
      // 在 Java 里可以这么用，当 left + right 溢出的时候，无符号右移保证结果依然正确
      int mid = (left + right + 1) >>> 1;

      int cnt = 0;
      for (int num : nums) {
        if (num < mid) {
          cnt += 1;
        }
      }

      // 根据抽屉原理，严格小于 4 的数的个数如果大于等于 4 个，
      // 此时重复元素一定出现在 [1, 3] 区间里

      if (cnt >= mid) {
        // 重复的元素一定出现在 [left, mid - 1] 区间里
        right = mid - 1;
      } else {
        // if 分析正确了以后，else 搜索的区间就是 if 的反面
        // [mid, right]
        // 注意：此时需要调整中位数的取法为上取整
        left = mid;
      }
    }
    return left;


  }

  /****300. 最长递增子序列
   * dp[i]表示以i结尾的最长递增子序列的长度
   * dp[i] = max(dp[j]) + 1 (j < i && nums[i] > nums[j])
   * *****/
  public int lengthOfLIS(int[] nums) {
    int len = nums.length;
    if(len < 2){
      return len;
    }
    int[] dp = new int[len];
    dp[0] = 1;
    int res = 0;
    for(int i = 1; i<len; i++){
      int max = 0;
      for(int j = 0; j<i; j++){
        if(nums[i] > nums[j]){
          max = Math.max(dp[j], max);
        }
      }
      dp[i] = max + 1;
      res = Math.max(res, dp[i]);
    }
    return res;

  }

  /***121. 买卖股票的最佳时机
   * 暴力解法： 两次遍历
   * dp[i] 表示当天能获得的最大利益，dp[i] = max(第i-1天的最大收益， 第i天的价格-前i-1天的最小值)
   * *****/
  public int maxProfit1(int[] prices) {
    int n = prices.length;
    int[] dp = new int[n];
    dp[0] = 0;
    int min = prices[0];
    int res = 0;
    for(int i = 1; i<n;i++){
      dp[i] = Math.max(prices[i] - min, dp[i-1]);
      min = Math.min(prices[i], min);
      res = Math.max(res, dp[i]);
    }

    return res;

  }


  /****309. 最佳买卖股票时机含冷冻期
   * 数组降维：买 卖 冷冻将维成两个维度，持有股票和不持有股票
   * 持有股票：今天买入的，之前买入的没有卖出
   * 不持有：今天或之前卖出，或者冷冻期
   * hold[i]：当前持有股票的最大收益=max( 昨天持有股票的收益，和昨天不持有股票的收益-今天买入)
   * 冷冻期：此如果今天要购买的话，那么只能是选择昨天没有卖出股票的情况，
   * notHold[i] ：当前不持有股票的最大收益= max(昨天不持有股票的收益，昨天持有股票+今天卖出)
   *
   * ****/
  public int maxProfit(int[] prices) {
    int n = prices.length;
    if(n == 0) return 0;
    int[] hold = new int[n];
    int[] notHold = new int[n];
    hold[0] = -prices[0];
    for(int i = 0; i<n; i++){
      if(i > 2){
        hold[i]= Math.max(hold[i-1], notHold[i-2] - prices[i]);
      } else {
        hold[i]=Math.max(hold[i-1],-prices[i]);
      }
      notHold[i]=Math.max(notHold[i-1],hold[i-1]+prices[i]);

    }
    return notHold[n-1];

  }

  /****322. 零钱兑换
   * f(n) = min(f(n-c1), f(n-c2), ... f(n-cn)) +1;
   * f(n)表示n大小的总和最少需要多少个硬币
   * 注意：凑不出的情况
   * ****/
  public int coinChange(int[] coins, int amount) {

    int[] dp = new int[amount+1];
    Arrays.fill(dp, amount + 1); // 要对数组进行填充，这是不可能存在的情况，遍历后如果凑不出，那么就不会更新，所以最后判断

    dp[0] = 0;
    for(int i = 1; i<=amount; i++){
      for(int c: coins){
        if(c <= i){
          dp[i] = Math.min(dp[i], dp[i-c] + 1);//双重循环实现上述的公式
        }
      }

    }
    return dp[amount]>amount? -1:dp[amount]; //首先要判断，如果dp[amount]>amount, 说明，

  }

  /****337. 打家劫舍 III
   * 抢劫该节点时，左子树和右子树节点不可抢劫，但是左右子树的左右子树可抢，计算其和
   * 不抢劫该节点时，抢劫左子树节点和右子树节点的和
   * ******/
  //直接递归,会有很多重复运算，所以耗时较长
  public int rob1(TreeNode root) {
    if(root == null){
      return 0;
    }
    int left = rob1(root.left);
    int right = rob1(root.right);
    int val = 0;
    if(root.left != null){
      val += rob1(root.left.right) + rob1(root.left.left);
    }
    if(root.right != null){
      val += rob1(root.right.left) + rob1(root.right.right) ;
    }
    return Math.max(right+left, val+root.val);


  }
  //用动态规划来减少运算，数组res[] 有两个元素，res[0]表示不抢劫该节点，res[1]表示抢劫该节点
  public int rob(TreeNode root) {
    int[] res = helper(root);
    return Math.max(res[0], res[1]);


  }
  private int[] helper(TreeNode root){
    if(root == null) return new int[2];
    int[] left = helper(root.left);
    int[] right = helper(root.right);
    int[] res = new int[2];
    res[0] = Math.max(left[1], left[0]) +Math.max(right[1], right[0]); //不抢劫该节点，抢劫左右子树的和，左右子树可以随便抢
    res[1] = root.val + left[0] + right[0];//抢劫该节点的话，则不抢劫左右节点
    return res;

  }

  /****338. 比特位计数
   * i是奇数：
   * i是偶数：
   * ****/
  public int[] countBits(int num) {
    int[] dp = new int[num+1];
    dp[0] = 0;
    for(int i = 1; i<= num; i++){
       if(i%2 == 0){
         dp[i] = dp[i/2];
       } else {
         dp[i] = dp[i-1]+1;
       }
    }
    return dp;
  }

  /***347. 前 K 个高频元素
   * ****/
  public int[] topKFrequent(int[] nums, int k) {
    HashMap<Integer, Integer> hashMap = new HashMap<>();
    int len = nums.length;
    int[] res = new int[k];
    for(int n:nums){
      hashMap.put(n, hashMap.getOrDefault(n, 0)+1);
//      if(hashMap.containsKey(n)){
//        int v = hashMap.get(n);
//        hashMap.replace(n, v+1);
//      } else{
//        hashMap.put(n, 1);
//      }
    }
    //return hashMap.entrySet().stream().sorted((m1,m2) -> m2.getValue() - m1.getValue()).limit(k).mapToInt(Map.Entry::getKey).toArray();
    List<Map.Entry<Integer, Integer>> list = new ArrayList<>(hashMap.entrySet());
    list.sort((m1, m2) -> m2.getValue() - m1.getValue());
    for(int i = 0; i<k; i++){
      res[i] = list.get(i).getKey();
    }
    return res;
  }

  /****394. 字符串解码
   * 辅助栈的方法:两个栈两个变量
   * 递归问题：结构和原问题一致的子问题，'[' 与']'作为递归的起始和终止条件
   * ***/
  public String decodeString(String s) {
    StringBuilder res = new StringBuilder();
    Deque<Integer> cntStk = new LinkedList<>();
    Deque<String> charStk = new LinkedList<>();
    int multi = 0;

    for(char c: s.toCharArray()){
      if(c=='['){ // 数量和字符串都入栈，并清空res变量和multi变量
        cntStk.push(multi);
        charStk.push(res.toString());
        multi = 0;
        res = new StringBuilder();

      } else if(c == ']'){ // 从两个栈中构建构建新的字符，并出栈
        StringBuilder temp = new StringBuilder();
        int cnt =cntStk.isEmpty()? 1:cntStk.pop();
        for(int i = 0; i<cnt; i++){
          temp.append(res);
        }
        res = new StringBuilder(charStk.peek() + temp); //构建新的res
        String str = charStk.pop();

      } else if(c >= '0' && c <= '9'){ //将字符转化为数字
        multi = multi * 10 + (c - '0');

      } else {
        res.append(c);

      }
    }
    return res.toString();

  }
  public String decodeString1(String s) {
   return dfsDecode(s, 0);

  }
  private int numberIndex = 0;// 记录for循环中i的位置
   private String dfsDecode(String s, int index){
    StringBuilder res = new StringBuilder();
    int multi = 0;
    for(int i= index;i < s.length(); i++){
      if(s.charAt(i) >= '0' && s.charAt(i) <= '9'){
        multi = multi * 10 + s.charAt(i) - '0';
      } else if(s.charAt(i) == '['){
        String str = dfsDecode(s, i+1);
        for(int j = 0; j<multi; j++){
          res.append(str);
        }
        multi = 0;
        i = numberIndex;

      } else if(s.charAt(i) == ']'){
        numberIndex = i;
        return res.toString();

      } else {
        res.append(s.charAt(i));
      }

    }
    return res.toString();
   }

   /****990. 等式方程的可满足性
    * 并查集：管理不相交集合，支持合并（两个不相交的集合合并在一起）和查询（查询两个元素是否在同一个集合）操作，
    * 例如：亲戚问题
    * *****/
   public boolean equationsPossible(String[] equations) {
     UnionFind unionFind = new UnionFind(26);
     for(String s : equations){
       char[] chars = s.toCharArray();
       if(chars[1] == '='){
         int x = chars[0] - 'a';
         int y = chars[3] - 'a';
         unionFind.union(x,y); //先将等式两边的元素放入一个集合
       }
     }
     for(String s: equations){
       char[] chars = s.toCharArray();
       if (chars[1] == '!') {
         int x = chars[0] - 'a';
         int y = chars[3] - 'a';
         if(unionFind.isConnected(x, y)){ // 查找不等式两边的元素是否在同一集合中，如果在，则与不等式矛盾
           return false;
         }
       }
     }
     return true;


   }

   //并查集的具体实现
   private class UnionFind{
     private int[] parent; //存储每个元素的父结点
     public UnionFind(int n ){
       parent = new int[n];
       for(int i = 0; i<n; i++){
         parent[i] = i; //表示每个元素是一个单独的集合
       }
     }
     //查询元素的父节点
     public int find(int x){
       if(parent[x] == x){
         return x;
       } else {
         parent[x] = find(parent[x]); //路径压缩
         return parent[x];
       }

//       while(parent[x] != x ){
//         parent[x] = parent[parent[x]]; //路径压缩
//         x = parent[x];
//       }
//       return x;
     }
     public void union(int x, int y){
       int rootX = find(x);
       int rootY = find(y);
       parent[rootX] = rootY; // 将x的父结点设为y
     }

     public boolean isConnected(int x, int y){
       return find(x) == find(y);
     }
   }


   /*****399. 除法求值
    * 并查集：带权值的并查集
    *******/
   public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
     int equationSize = equations.size();
     UnionFind2 unionFind2 = new UnionFind2(equationSize * 2);
     // 第一步 进行合并，将string 映射成int
     HashMap<String, Integer> hashMap = new HashMap<>(equationSize * 2);
     int id = 0;
     for(int i = 0; i<equationSize; i++) {
       List<String> equation = equations.get(i);
       String val1 = equation.get(0);
       String val2 = equation.get(1);
       if (!hashMap.containsKey(val1)) {
         hashMap.put(val1, id);
         id++;
       }
       if (!hashMap.containsKey(val2)) {
         hashMap.put(val2, id);
         id++;
       }
       unionFind2.union(hashMap.get(val1), hashMap.get(val2), values[i]);
     }

       //第二步：进行查询
       int queriesSize = queries.size();
       double[] res = new double[queriesSize];
       for(int j = 0; j < queriesSize; j++){
         String var1 = queries.get(j).get(0);
         String var2 = queries.get(j).get(1);
         Integer id1 = hashMap.get(var1);
         Integer id2 = hashMap.get(var2);
         if(id1 == null || id2 == null){
           res[j] = -1.0d;
         } else {
           res[j] = unionFind2.isConnected(id1, id2);

         }
       }
     return res;

   }


   private class UnionFind2{
     private int[] parent;
     private double[] weight;//指向父节点的权重
     public UnionFind2(int n){
       parent = new int[n];
       weight = new double[n];
       for(int i = 0; i<n; i++){
         parent[i] = i;
         weight[i] = 1.0d;
       }
     }

     public void union(int x, int y, double val){
       int rootX = find(x);
       int rootY = find(y);
       if(rootX == rootY){
         return;
       }
       parent[rootX] = rootY;
       weight[rootX] = weight[y] * val / weight[x]; //weight的更新是重点

     }

     public int find(int x){
       if(parent[x] == x){
         return x;
       } else {
         int pre = parent[x];
         parent[x] = find(parent[x]);
         weight[x] *= weight[pre];//所有父节点的权重都要相乘
         return parent[x];
       }
     }
     public double isConnected(int x, int y){
       int rootX = find(x);
       int rootY = find(y);
       if(rootX == rootY){
         return weight[x] /weight[y];
       } else {
         return -1d;
       }
     }
   }

   /****416. 分割等和子集
    * 0-1背包问题
    * 从这个数组中挑出一些数字，使得这些数字的和为数组元素总和的一半
    * dp[i][j]表示从0-i区间中找到一个选取一些正整数，正整数的和为j
    * 不选择当前元素 dp[i][j] = dp[i-1][j]，说明在0-i-1这个区间内已经存在一些数字之和为j
    * 选择当前元素。0 - i-1 区间内得找到一些数字之和为 j - nums[i], dp[i][j] = dp[i-1][j-nums[i]];
    *
    *****/
   public boolean canPartition(int[] nums) {
     int len = nums.length;
     if(len == 0){
       return false;
     }
     int sum = 0;
     for (int num : nums) {
       sum += num;
     }
     if(sum%2 != 0){
       return false;
     }
     int target = sum/2;
     boolean[][] dp = new boolean[len][target + 1];
     if(nums[0] <= target){
       dp[0][nums[0]] = true; // 第一个数只能填满自己容积的背包
     }
     for(int i = 1; i<len; i++){
       for(int j = 0; i <= target; j++){
         dp[i][j] = dp[i - 1][j];
         if(nums[i] == j){
           dp[i][j] = true;
           continue;
         }
         if(nums[i] < j){
           dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
         }
       }
     }
     return dp[len - 1][target];


   }

   /****448 找到所有数组中消失的数字
    * ****/
   public List<Integer> findDisappearedNumbers(int[] nums) {
     List<Integer> res = new ArrayList<>();
     int[] cnt = new int[nums.length];
     for(int i = 0; i<nums.length; i++){
       cnt[nums[i] - 1] += 1;
     }

     for (int i = 0; i < cnt.length; i++) {
       if(cnt[i] == 0){
         res.add(i + 1);
       }
     }
     return res;
   }
  //将索引i对应的num[i] 对应的索引的值置为负数，表示该索引值已经出现过了, 最后为数组中为正数的索引值就是没有出现过的
  public List<Integer> findDisappearedNumbers1(int[] nums) {
    List<Integer> res = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
      if(nums[Math.abs(nums[i]) - 1 ]>0 ){ //如果该位置上已经是负数了，说明该索引值已经出现了，不要再置反了
        nums[Math.abs(nums[i]) - 1] = -nums[Math.abs(nums[i]) - 1];

      }
    }

    for (int i = 0; i < nums.length; i++) {
      if(nums[i] > 0){
        res.add(i + 1);
      }
    }
    return res;
  }


  //桶排序的思想
  public List<Integer> findDisappearedNumbers2(int[] nums) {
    List<Integer> res = new ArrayList<>();
    //将num[i] 放到nums[nums[i] - 1], 让nums[i] = i+1
    for (int i = 0; i < nums.length; i++) {
      while(nums[i] != i+1 && nums[i] !=  nums[nums[i]-1]) swap(nums, i, nums[i] -1);  // while循环中的第二个判断条件很重要
    }
    for (int i = 0; i < nums.length; i++) {
      if(nums[i] != i + 1){
        res.add(i+1);
      }
    }
    return res;

  }





    /****461. 汉明距离
     * 异或的运算：两个二进制树输入位不同时为1，计算两个值异或后数值中1的个数
     * *****/
   public int hammingDistance(int x, int y) {
    // return Integer.bitCount(x^y); // 内置的位计数算法
     int xor = x^y;
     int distance = 0;
     while(xor != 0){
       if(xor%2== 1){
         distance ++;
       }
       xor= xor>>1;
     }
     return distance;
   }

   /***543. 二叉树的直径
    * ****/
   int max = 0;
   public int diameterOfBinaryTree(TreeNode root) {
    depth(root);
    return max;

   }
   private int depth(TreeNode root){
     if(root == null){
       return 0;
     }
     int left = depth(root.left);
     int right = depth(root.right);
     max = Math.max(max, left+right);
     return Math.max(left, right)+1;
   }


   /*****617. 合并二叉树
    * *****/
   public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
     if(t1 == null){
       return t2;
     }
     if(t2 == null){
       return t1;
     }
     t1.val = t1.val+t2.val;
     t1.left = mergeTrees(t1.left, t2.left);
     t1.right = mergeTrees(t1.right, t2.right);
     return t1;

   }


   /***207. 课程表
    * 拓扑排序 + 入度表
    * ***/
   public boolean canFinish(int numCourses, int[][] prerequisites) {
     int[] indegrees = new int[numCourses]; // 记录一个课程的入度
     List<List<Integer>> adjacency = new ArrayList<>(); //表示一个课程，是哪些课程的前驱课程
     Queue<Integer> queue = new LinkedList<>(); //如何一个节点变成了没有任何入边的节点，那么可以加入学习队列
     for (int i = 0; i < numCourses; i++) {
       adjacency.add(new ArrayList<>());
     }
     for(int[] cp: prerequisites){
       indegrees[cp[0]]++;
       adjacency.get(cp[1]).add(cp[0]);
     }
     for (int i = 0; i < numCourses; i++) {
       if(indegrees[i] == 0) {
         queue.add(i);
       }
     }
     while(!queue.isEmpty()){
       int pre = queue.poll();
       numCourses--;
       for(int cur : adjacency.get(pre)){
         if(--indegrees[cur] == 0){
           queue.add(cur);
         }
       }
     }
     return numCourses == 0;

   }


   /****6. Z 字形变换
    * 每一行都是一个StringBuilder,最后再组合
    * 从上往下或者从下往上添加数据时，遇到首行或者尾行时，需要调整方向
    * ****/
   public String convert(String s, int numRows) {
     if(numRows < 2) return s;

     List<StringBuilder> list = new ArrayList<>();
     for (int i = 0; i < numRows; i++) {
       list.add(new StringBuilder());
     }
     int i = 0;
     int flag = -1;
     for (int j = 0; j < s.toCharArray().length; j++) {
       list.get(i).append(s.charAt(j));
       if( i==0 || i== numRows -1) flag = -flag;
         i += flag;

     }
     StringBuilder res = new StringBuilder();
     for (StringBuilder sb : list) {
       res.append(sb);
     }
     return res.toString();


   }

   /***7. 整数反转
    * ***/
   public int reverse(int x) {
     String str = String.valueOf(x);
     StringBuilder sb ;
    if(x < 0){
      try{
       sb = new StringBuilder(str.substring(1));
        int res = Integer.parseInt(sb.reverse().toString());
        return 0-res;
      }catch (NumberFormatException e){
        return 0;
      }

    } else {
      try{
        sb = new StringBuilder(str);
        int res = Integer.parseInt(sb.reverse().toString());
        return res;
      }catch (NumberFormatException e){
        return 0;
      }
    }

   }
  public int reverse1(int x) {
     int res = 0;
     while(x != 0){
       int pop = x%10;
       if(res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE && pop > 7)){
         return 0;
       }
       if(res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE && pop < 8)){
         return 0;
       }

       res = res*10 + pop;
       x = x/10;

     }
     return res;
  }

  /*****8. 字符串转换整数 (atoi)
   * ****/
  public int myAtoi(String s) {
    if(s.length() == 0){
      return 0;
    }
    s = s.trim();
    boolean flag = true;
    if(s.charAt(0) == '-'){
      flag = false;
      s = s.substring(1);
    } else if(s.charAt(0)=='+'){
     s =  s.substring(1);
    }
    int res = 0;
    int index  = 0;
    while(index < s.length() ){
      if(s.charAt(index) <'0' || s.charAt(index) > '9'){
        break;
      }
      if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && (s.charAt(index) - '0') > Integer.MAX_VALUE % 10)) {
        return Integer.MAX_VALUE;
      }
      if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && (s.charAt(index) - '0') > -(Integer.MIN_VALUE % 10))) {
        return Integer.MIN_VALUE;
      }

      if(flag){
        res = res*10 + s.charAt(index);
      } else {
        res = res*10 - s.charAt(index);
      }
      index++;

    }
    return res;



//    StringBuilder sb = new StringBuilder();
//    for (int i = 0; i < s.toCharArray().length; i++) {
//      if(Character.isDigit(s.charAt(i))){
//        sb.append(s.charAt(i));
//      } else{
//        break;
//      }
//    }
//    if(sb.length() == 0){
//      return 0;
//    }
//    if(flag){
//      try{
//        int res = Integer.parseInt(sb.toString());
//        return res;
//      } catch (NumberFormatException e){
//        return Integer.MAX_VALUE;
//      }
//
//    } else{
//      try{
//        int res = Integer.parseInt(sb.toString());
//        return 0-res;
//      } catch (NumberFormatException e){
//        return Integer.MIN_VALUE;
//      }
//    }

  }

  /***9. 回文数
   ***/
  public boolean isPalindrome(int x) {
    if(x < 0){
      return false;
    }
    int r = 0;
    int temp = x;
    while(temp != 0){
      int pop = temp%10;
      r = r * 10 + pop;
      temp = temp/10;
    }
    if(r == x){
      return true;
    } else{
      return false;
    }

  }



  public String solution3(String s){
    int index = 0;
    for (int i = 0; i < s.toCharArray().length; i++) {
      if(s.charAt(i) == '?'){
        index = i;
      }
    }
    int a1=0, b1=0, a2=0, b2=0;
    for(int i = 0; i< index; i++){
      if(s.charAt(i) == 'a'){
        a1+=1;
      } else if(s.charAt(i) == 'b'){
        b1 +=1;
      }
    }
    if(index >= 2 && s.charAt(index-1) == 'a' &&s.charAt(index-2) =='a'){
      a1 = 2;
    }else if(index >=2 &&s.charAt(index-1) == 'b' &&s.charAt(index-2) =='b'){
      b1 = 2;
    } else if(index < s.length() - 2 && s.charAt(index+1) == 'a' &&s.charAt(index+2) =='a'){
      a2 = 2;
    } else if(index < s.length() - 2 && s.charAt(index+1) == 'b' &&s.charAt(index+2) =='b') {
      b2 = 2;
    } else if(index < 2){

    }
    String s1;
    if(a1 == 2 || a2 == 2 ){
      s1 =  s.substring(0, index)+'b'+s.substring(index+1);
      return solution3(s1);

    } else if( b1 == 2|| b2 == 2){
      s1 =  s.substring(0, index)+'a'+s.substring(index+1);
      return solution3(s1);

    } else {
      s1 =  s.substring(0, index)+'a'+s.substring(index+1);
      return solution3(s1);
    }


}
  public int solution2(int[] A){
    if(A == null || A.length == 0){
      return 0;
    }
    int[] dp = new int[A.length];
    dp[0] = 0;
    for(int i = 1; i< A.length; i++){
      dp[i] = A[i]+A[i-1];
    }
    //找到dp中不相邻出现且出现次数最多的数字
    int flag = 0; //上一次出现的值
    int index = -1; //上上一个相同值的索引

    int res = 0;

    int[] bucktet = new int[100000];
    int cnt = 0;
    for (int i = 0; i < dp.length; i++) {
      if(flag != dp[i]){
        bucktet[dp[i]]++;
      } else { //和上一个相等的时候，需要判断是否是连着的两个
        if (index  == i ) {
          bucktet[dp[i]]++;
        } else {
          index = i + 1 ;
        }
      }
      flag = dp[i]; //记录上一次的值

      res = Math.max(bucktet[dp[i]], res);
    }

    return res;

  }

  public int solution23(int[] A){
    if(A == null || A.length == 0){
      return 0;
    }
    int[] dp = new int[A.length];
    dp[0] = 0;
    for(int i = 1; i< A.length; i++){
      dp[i] = A[i]+A[i-1];
    }
    //找到dp中不相邻出现且出现次数最多的数字
    int prev = -1; //上一次出现的值
    boolean flag = true; // 表示上一个相同的是否添加过了，上一个添加过了则为true,就不再添加了
    int res = 0;

    int[] bucktet = new int[100000];
    int cnt = 0;
    for (int i = 0; i < dp.length; i++) {
      if(dp[i] != prev){
        bucktet[dp[i]]++;
      } else { //和上一个相等的时候，需要判断是否是连着的两个
        if(!flag){ // 表示上一个相同的没添加过
          bucktet[dp[i]]++;
          flag = true; //添加后要修改变量
        } else{ //上一个添加过了，这个则不添加，并且这个的下一个可以再添加，所以置为了false
          flag = false;
        }
      }
      prev = dp[i]; //记录上一次的值

      res = Math.max(bucktet[dp[i]], res);
    }

    return res;

  }
  public int solution24(int[] A){
    if(A == null || A.length == 0){
      return 0;
    }
    int[] dp = new int[A.length];
    dp[0] = -1;
    for(int i = 1; i< A.length; i++){
      dp[i] = A[i]+A[i-1];
    }
    //找到dp中不相邻出现且出现次数最多的数字

    int res = 0;

    int[] bucktet = new int[100000];
    int cnt = 0;
    for (int i = 0; i < dp.length; i++) {
      if(dp[i] == -1){
        continue;
      }
      bucktet[dp[i]]++;
      if( i+1 < dp.length && dp[i+1] == dp[i]){
        i++;
      }

      res = Math.max(bucktet[dp[i]], res);
    }

    return res;

  }
  public int solution222(int[] A){
    if(A == null || A.length == 0){
      return 0;
    }
    int[] dp = new int[A.length];
    dp[0] = 0;
    for(int i = 1; i< A.length; i++){
      dp[i] = A[i]+A[i-1];
    }
    //找到dp中不相邻出现且出现次数最多的数字
    int pre = 0; //上一次出现的值

    int res = 0;
    int cnt = 1; //连续出现的次数
    int[] bucktet = new int[100000];

    for (int i = 1; i < dp.length; i++) {
      if(dp[i] != pre){
        bucktet[dp[i]]++;
        if(cnt >= 2){
          bucktet[pre] = (cnt+1)/2;
        }
        cnt = 1;
      } else { //和上一个相等的时候，需要判断是否是连着的两个
        cnt +=1;
        if(i == dp.length -1){
          bucktet[dp[i]] = (cnt+1)/2;
        }
      }
      pre = dp[i]; //记录上一次的值

      res = Math.max(bucktet[dp[i]], res);
    }

    return res;

  }

  public static int solution1(int[] A){
    int res= 0;
    int max =A[0];
    for(int i =0;i<A.length;i++){
      max = Math.max(max,A[i]);
    }
    int []count = new int[1000010];
    for(int i =0;i<A.length;i++){
      if(A[i]==max){
        count[A[i]]=1;
      } else {
        count[A[i]]=count[A[i]]+1;
      }
    }
    for(int i =0;i<count.length;i++){
      if(count[i]>=2)res+=2;
      if(count[i]==1)res++;
    }
    return res;
  }

  public int[] solution22(String[] cars){
    if(cars == null||cars.length == 0){
      return null;
    }
    int[] res = new int[cars.length];
    for(int i = 0; i<cars.length; i++){
      for(int j = i + 1; j<cars.length; j++){
        int x = Integer.parseInt(cars[i], 2);
        int y = Integer.parseInt(cars[j], 2);
        int temp = Integer.bitCount(x^y);
        if(temp <= 1){
          res[i]++;
          res[j]++;
        }

      }
    }
    return res;

  }



  public static int[] solution2(String[] cars) {
    int [] res = new int[cars.length];

    int []table = new int[]{32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,1,0} ;

    int []numArray = new int[cars.length];
    for(int i =0;i<cars.length;i++){
      numArray[i]=Integer.parseInt(cars[i],2);
    }
    HashMap<Integer,Integer> map =new HashMap<>();

    for(int i =0;i<numArray.length;i++){
      if(map.containsKey(numArray[i])){
        map.put(numArray[i],map.get(numArray[i])+1);
      }else{
        map.put(numArray[i],1);
      }
    }

    for(int i =0;i<numArray.length;i++){
      int tempNum = numArray[i];
      while(tempNum>0){
        for(int k=0;k<table.length;k++){
          if(table[k]==0){
            res[i]+=map.get(numArray[i])-1;
            continue;
          }
          if(tempNum>=table[k]){
            tempNum-=table[k];
            if(map.containsKey(numArray[i]-table[k])){
              res[i]+=map.get(numArray[i]-table[k]);
            }
          }else{
            if(map.containsKey(numArray[i]+table[k])){
              res[i]+=map.get(numArray[i]+table[k]);
            }
          }
        }
      }
    }

    return res;
  }



  /****12. 整数转罗马数字
   * 尽可能使用较大字符来对应数字，使得数字的长度最短
   * ***/
  public String intToRoman(int num) {
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] str = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX","V", "IV", "I"};
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      while(num >= values[i]){
        num -= values[i];
        sb.append(str[i]);
      }

    }
    return sb.toString();
  }

  /****13. 罗马数字转整数
   ***/
  public int romanToInt(String s) {
    int res = 0;
    int preNum = getValue(s.charAt(0));
    for (int i = 1; i < s.toCharArray().length; i++) {
      int num = getValue(s.charAt(i));
      if(preNum < num ){
        res -= preNum;
      } else {
        res+=preNum;
      }
      preNum = num;
    }
    res += preNum;
    return res;
   }
  private int getValue(char ch) {
    switch(ch) {
      case 'I': return 1;
      case 'V': return 5;
      case 'X': return 10;
      case 'L': return 50;
      case 'C': return 100;
      case 'D': return 500;
      case 'M': return 1000;
      default: return 0;
    }
  }

  /****14. 最长公共前缀
   * 横行对比
   * 纵向对比：比较字符串中每一个字符是否相等
   ***/
  public String longestCommonPrefix(String[] strs) {
    if(strs == null || strs.length == 0){
      return " ";
    }
    String prefix = strs[0];
    for(int i = 0; i<prefix.length(); i++){
      char ch = prefix.charAt(i);
      for (int j = 0; j < strs.length; j++) {
        if(strs[j].charAt(i) != ch || i == strs[j].length()){
          return prefix.substring(0, i);
        }

      }
    }
    return prefix;
  }

  /****16. 最接近的三数之和
   * *****/
//  public int threeSumClosest(int[] nums, int target) {
//
//  }



  /***989. 数组形式的整数加法
   * ***/
  public List<Integer> addToArrayForm(int[] A, int K) {
    List<Integer> res = new ArrayList<>();
    int sum = 0;
    int carry = 0;
    int n = A.length;
    int i = n-1;
    while( i>=0 || K >0){
      int cur1 = i>=0? A[i]:0;
      int cur2 = K != 0? K%10:0;
      sum = cur1 + cur2 + carry;

      carry = sum/10;
      K = K/10;
      i--;
      res.add(0, sum);
    }
    if(carry != 0){
      res.add(0, carry);
    }
    return res;
  }

  /***28. 实现 strStr()
   * *****/
  public int strStr(String haystack, String needle) {
    if(haystack.length() == 0 || needle.length() == 0){
      return 0;
    }

  }





    public static void main(String[] args){
    System.out.println("hello top100");
    Top100 top100 = new Top100();
    //int s = top100.lengthOfLongestSubstring("abcabcbb");
    int[] n = {0,1,3,1,2,2,1,0,4};
    int[] m = {9,9,9,9,9,9};
    int[] m1 = {5,3,1,3,2,3};
    int[] m2 = {9,0,9,0,9,0};
    String[] s = {"100","110","010","011","100"};
    String[] ss = {"0011","0111","0111","0110","0000"};
      int res = top100.reverse(-123);
      int r = top100.myAtoi("  -42");
      //String s2 = top100.solution3("??abb");
      int rr = top100.solution24(m);
      boolean rrr = top100.isValid2("()");
      System.out.println(rr);
      //int[] nums = top100.solution22(s);
//      int[] nums = top100.solution2(s);
//      for (int num : nums) {
//        System.out.println(num);
//
//      }


  }
}
