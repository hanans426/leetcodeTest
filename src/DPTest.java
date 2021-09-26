/**
 * User: gaohan
 * Date: 2020/12/22
 * Time: 17:24
 *
 */
public class DPTest {

  /******70. 爬楼梯
   * 典型的动态规划
   * dp[i] = dp[i-1] + dp[i-2]
   * ****/
  public int climbStairs(int n) {
    if(n <= 2){
      return n;
    }
    int pre2 = 1; int pre1 = 2;
    for(int i = 3; i<=n; i++){
      int cur = pre1 + pre2;
      pre2 = pre1;
      pre1 = cur;
    }
    return pre1;

  }

  /****198. 打家劫舍
   * 如果偷盗第k间房屋，那么偷盗金额就是前k-2 间总金额加上k房间内的金额
   * 如果不偷盗第k间房屋，那么偷盗金额就是前k-1间总金额
   * 所以，k个房间内的偷盗金额是个动态规划的过程，dp[k] = max(dp[k-2]+num[k], dp[k-1])
   * ***/
  public int rob(int[] nums) {
    int pre2 = 0;// 前k-2 个偷盗总金额，
    int pre1 = 0; //前k-1个偷盗总金额
    for(int i = 0; i<nums.length; i++){
      int cur = Math.max(pre2 + nums[i], pre1);
      pre2 = pre1;
      pre1 = cur;
    }
    return pre1;

  }
  //动态规划的普通写法
  public int rob1(int[] nums){

    if(nums.length == 1){
      return nums[0];
    } else if(nums.length == 0){
      return 0;
    }
    int[] mem = new int[nums.length];
    mem[0] = nums[0];
    mem[1] = Math.max(nums[0], nums[1]);
    for(int i = 2; i<nums.length; i++){
      mem[i] = Math.max(mem[i-2] + nums[i], mem[i-1]);
    }
    return mem[nums.length-1];
  }

  /****213. 打家劫舍 II
   * *****/
  public int rob2(int[] nums) {

    int n = nums.length;
    if(n==0 ||nums==null){
      return 0;
    }
    if(n == 1){
      return nums[0];
    }

    return Math.max(robb(nums, 0, n-2), robb(nums, 1, n-1));

  }
  private int robb(int[] nums, int first, int last){
    int pre2 = 0;
    int pre1 = 0;
    int cur = 0;
    for(int i = first; i<=last; i++){
      cur = Math.max(pre2 + nums[i], pre1);
      pre2 = pre1;
      pre1 = cur;
    }

    return pre1;
  }

  /******
   * **信件错排**
   * 题目描述：有 N 个 信 和 信封，它们被打乱，求错误装信方式的数量。
   * 定义一个数组 dp 存储错误方式数量，dp[i] 表示前 i 个信和信封的错误方式数量。假设第 i 个信装到第 j 个信封里面，而第 j 个信装到第 k 个信封里面。根据 i 和 k 是否相等，有两种情况：
   * - i == k，交换 i 和 k 的信后，它们的信和信封在正确的位置，但是其余 i-2 封信有 dp[i-2] 种错误装信的方式。由于 j 有 i-1 种取值，因此共有 (i-1)\*dp[i-2] 种错误装信方式。
   * - i != k，交换 i 和 j 的信后，第 i 个信和信封在正确的位置，其余 i-1 封信有 dp[i-1] 种错误装信方式。由于 j 有 i-1 种取值，因此共有 (i-1)\*dp[i-1] 种错误装信方式。
   * ****/

  private int envelope(int n){
    if(n == 0 || n == 1){
      return 0;
    }
    int[] dp = new int[n];
    dp[0] = 0;
    dp[1] = 0;
    dp[2] = 1;
    for(int i = 3; i < n; i++){
      dp[i] = (i-1) * dp[i-2] + (i-1)*dp[i-1];
     }
     return dp[n-1];
  }

  /***64. 最小路径和
   * dp[i][j] = min(dp[i-1][j] + grid[i][j], dp[i][j-1] + grid[i][j]
   * *****/
  //二维数组进行动态规划
  // todo 可以利用一维数组进行空间和时间上的优化
  public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    if (grid == null || m < 1 || grid[0] == null || n < 1) {
      return 0;
    }
    int[][] dp = new int[m][n];
    dp[0][0] = grid[0][0];

    for(int i = 1; i<n; i++){
      dp[0][i] = dp[0][i-1] + grid[0][i];
    }

    for(int i = 1; i<m; i++){
      dp[i][0] = dp[i-1][0] + grid[i][0];
    }

    for(int i = 1; i<m; i++){
      for(int j = 1; j<n; j++){
        dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1])+ grid[i][j];

      }
    }

    return dp[m-1][n-1];

  }

 /****62. 不同路径
  * *****/
 //动态规划
 public int uniquePaths(int m, int n) {
   int[][] dp = new int[m][n];
   for(int i = 0; i < m; i++){
     for(int j = 0; j < n; j++){
       if (i == 0 || j == 0){
         dp[i][j] = 1;
       } else{
         dp[i][j] = dp[i-1][j]+dp[i][j-1];
       }

     }
   }

   return dp[m-1][n-1];
 }
 //排列组合的方法也可，总移动步数是S=m+n-2,向下的移动步数是D=m-1,那么就是从s中取出D的组合数量,排列组合的公式
  //todo 阶乘的算法
 public int uniquePaths1(int m, int n) {
   int S = m+n-2;
   int D = m-1;
   int res = 1;
   for(int i = 1; i < D; i++){
     res = res *(S-D+i)/i;
   }
   return res;
 }


 /***303. 区域和检索 - 数组不可变
  * ****/

 /****53. 最大子序和
  * dp[i] = max(dp[i-1]+nums[i], nums[i])
  * ****/
 public int maxSubArray(int[] nums) {
   if(nums == null ||nums.length == 0){
     return 0;
   }
   int[] dp = new int[nums.length];
   dp[0] = nums[0];
   int maxSum = dp[0];
   for(int i = 1; i<nums.length; i++){
     dp[i] = Math.max(dp[i-1]+nums[i], nums[i]);
     maxSum = Math.max(maxSum, dp[i]);
   }

   return maxSum;

 }

 /***413. 等差数列划分
  * dp[i]表示A中第i个元素结尾的等差数组的个数，如果满足A[i]与A[i-1]也是等差的话，那么dp[i] = dp[i-1]+1
  * ****/
 public int numberOfArithmeticSlices(int[] A) {
   if(A==null || A.length == 0){
     return 0;
   }
   int[] dp = new int[A.length];
   dp[0] = 0;
   int res = 0;
   for(int i = 0; i<A.length; i ++){
     if(A[i]-A[i-1] == A[i-1] - A[i-2]){
       dp[i] = dp[i-1] + 1;
     }
   }
   for(int num :dp){
     res += num;
   }
   return res;

 }

 /****343. 整数拆分
  * 动态规划
  * * ****/
 public int integerBreak(int n) {
    int[] dp = new int[n+1];
    dp[1] = 1;
    int  pre = 0;
    for (int i = 2; i<=n; i++){
      for(int j = 1;j <= i-1; j++ ) {
        dp[i] = Math.max(dp[i], Math.max(j*dp[i-j], j*(i-j)));

      }
    }
    return dp[n];
 }

 /*****279. 完全平方数
  * dp[i]表示正整数i 的完全平方数个数 dp[i] = min(dp[i], dp[i-j*j]+1)，重点是内部循环的限制条件和dp方程
  * 任何正整数都可以表示成不超过四个的完全平方和
  * todo 对动态规划的不太理解
  * *****/
 public int numSquares(int n) {
   int[] dp = new int[n+1];
   dp[0]=0;
   for(int i = 1; i<=n; i++){
     dp[i] = i; //最坏的情况
     for(int j = 1; i-j*j>=0; j++){ // i中有多少个可以用完全平方和表示的数
       dp[i] = Math.min(dp[i], dp[i-j*j]+1);
     }
   }
   return dp[n];

 }

 /*****91. 解码方法
  * 当前位数是否要去上一位进行合并res = join[i-1] + nojoin[i-1]
  * join[], nojoin[]
  *   - 如果和上一个进行合并，则必须上一位没有和上上一位合并才可以， 合并后小于等于26，如果当前位是0，则必须和上一位进行合并
  *   join[i] = nojoin[i-1]
  *   - 如果不和上一个进行合并,则不必考虑上一位的合并情况，
  *   nojoin[i] = nojoin[i-1] + join[i-1]
  * ******/
 public int numDecodings(String s) {
   int n = s.length();
   int[] nojoin = new int[n];
   int[] join = new int[n];

   int[] array = new int[n];

   for (int i = 0; i < array.length; i++) {
     array[i] = Integer.valueOf(s.charAt(i)+"");
   }
   if(array[0]==0||s.length()==0) {
     return 0;
   }
   nojoin[0] = 1;
   join[0] = 0;

   for(int i = 1; i < n; i++){
     nojoin[i] = nojoin[i-1] + join[i-1];
     join[i] = nojoin[i-1];

     if(array[i] == 0){
       nojoin[i] = 0;
     }

     if(array[i-1]*10 + array[i] > 26){
       join[i] = 0;
     }


   }

   return nojoin[n-1]+join[n-1];

 }



   public static void main(String[] args){
    System.out.println("hello DP");
    DPTest dpTest = new DPTest();
    int[] r = {1,2,3,1};
    dpTest.rob(r);
    dpTest.numDecodings("226");



  }
}
