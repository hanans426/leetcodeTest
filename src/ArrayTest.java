import jdk.internal.util.xml.impl.Pair;

import java.util.*;
import java.util.ArrayList;


/**
 * User: gaohan
 * Date: 2020/8/17
 * Time: 11:59
 * 数组
 */
public class ArrayTest {



/*** 1144. 递减元素使数组呈锯齿状 ***/
  public  int movesToMakeZigzag(int[] nums){
    if(nums.length <= 2){
      return 0;
    } else {
      return Math.min(compute(Arrays.copyOf(nums, nums.length), 1), compute(nums, 0));

    }
  }

  public int compute(int[] arr, int idx){
    int count = 0;
    for(; idx< arr.length; idx += 2){
      int cur = arr[idx];
      if(idx > 0 && arr[idx - 1] >= cur){
        count += arr[idx - 1] - cur + 1;
        arr[idx - 1] = cur - 1;
      }

      if(idx + 1 < arr.length && arr[idx + 1] >= cur){
        count += arr[idx + 1] - cur + 1;
        arr[idx + 1] = cur -1;
      }
    }
    return count;
  }


  /*** 剑指 Offer 53 - II. 0～n-1中缺失的数字 ***/
  //func1: 遍历法 要考虑到首位和末位缺失的情况
  public int missingNumber(int[] nums) {
    int miss = nums.length ;
    if(nums[0] == 1) return 0;
    for(int i = 0; i < nums.length; i++){
      if(nums[i] != i){
        miss = i;
      }

    }
    return miss;

  }

  //func2: 二分查找法, 返回得到右子数组首位元素对应的索引
  public int missingNumber2(int[] nums){
    int i = 0;
    int j = nums.length;
    while(i <= j){
      int m = (i+j)/2;
      if(nums[m] == m){
        i = m + 1;
      } else{
        j = m - 1;
      }
    }
    return i;
  }

  /*** 674. 最长连续递增序列 ***/
  public int findLengthOfLCIS(int[] nums){
    if(nums.length <= 1){
      return nums.length; //注意此处要判断数组长度为0或者1的情况
    } else {
      int ans = 1;
      int count = 1;
      for(int i = 0; i< nums.length - 1; i++){
        if(nums[i + 1] > nums[i]){
          count++;
        } else {
          count = 1;
        }
        ans = Math.max(ans, count);
      }
      return ans;
    }

  }

  /*** 867. 转置矩阵 ***/
  public int[][] transpose(int[][] A) {
    int len =  A.length;
    int width = A[0].length;
    int[][] res = new int[width][len];

    for(int i = 0; i < len; i++){
      for(int j = 0; j < width; j++){
        res[j][i] = A[i][j];
      }
    }
    return res;
  }

  /*** 867. 转置矩阵 ***/
  class SubrectangleQueries {

    private int[][] rectangle;

    public SubrectangleQueries(int[][] rectangle) {
      this.rectangle = rectangle;
    }

    public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
      int len = rectangle.length;
      int width = rectangle[0].length;
      //todo 判断row1 row2大小
      for(int i = row1; i < row2; i++){
        for(int j = col1; j < col2; j++){
          rectangle[i][j] = newValue;
        }
      }

    }

    public int getValue(int row, int col) {
      return rectangle[row][col];

    }
  }

  /***面试题 17.10. 主要元素***/
  public int majorityElement(int[] nums) {
    int len = nums.length;
    int temp = nums[0];
    int count = 1;
    int res;
    for(int i = 1; i < len; i++){
      if(nums[i] != temp){
       count--;
      } else {
        count++;
      }

      if(count == 0){
        count =1;
        temp = nums[i];
      }
    }
    if(count > 0){
      res = temp;
    } else {
      res = -1;
    }

    return res;


  }

  /***1535. 找出数组游戏的赢家***/

  public int getWinner(int[] arr, int k) {

    int temp = arr[0];
    int count = 0;
    for(int i = 0; i < arr.length && count < k; i++){
      if(temp > arr[i]){
        count++;

      } else {
        temp =  arr[i];
        count = 1;
      }

    }

    return temp;

  }

//  /***1552. 两球之间的磁力***
//   * 要求数组中m 个球中相邻两球距离的最小值
//   */
//
//  public int maxDistance(int[] position, int m) {
//
//  }
  /***1672. 最富有客户的资产总量****/

  public int maximumWealth(int[][] accounts) {
    int r = accounts.length;
    int c = accounts[0].length;
    int maxW = 0;
    for(int i = 0; i < r; i++){
      int sum = 0;
      for(int j = 0; j < c; j++){
        sum += accounts[i][j];
      }
      if(maxW < sum){
        maxW = sum;
      }

    }
    return maxW;
  }
  //TODO java8的流式接口是什么
//  public int maximumWealth1(int[][] accounts) {
//
//    return Arrays.stream(accounts).map(i -> Arrays.stream(i).sum()).max(Integer::compareTo).get();
//  }

  //求每行的第一个元素和其他所有元素的和，并将值赋给每行的第一个元素，然后每行元素和第一行元素和进行比较，并将较大值赋给第一行的第一个元素
  public int maximumWealth2(int[][] accounts) {
    int r = accounts.length;
    for(int i = 0; i < r; i++){
      for(int j = 1; j <accounts[i].length; j++){
        accounts[i][0] += accounts[i][j];
      }
      accounts[0][0] = Math.max(accounts[0][0], accounts[i][0]);


    }
    return accounts[0][0];
  }

  /***1480 给你一个数组 nums 。数组「动态和」的计算公式为：runningSum[i] = sum(nums[0]…nums[i])****/
  public int[] runningSum(int[] nums) {
    int[] sums = new int[nums.length];
    sums[0] = nums[0];
    for(int i = 1; i < nums.length; i++){
      sums[i] += sums[i-1] + nums[i];
    }
    return sums;
  }

  /***1470. 重新排列数组**
   *思路：找规律**/
  public int[] shuffle(int[] nums, int n){
    int[] ans = new int[2*n];
    for(int i = 0; i< n; i++){
      ans[2*i] = nums[i];
      ans[2*i + 1] = nums[i + n];
    }

    return ans;
  }

  /***1431. 拥有最多糖果的孩子***/
  // 注意边界条件，大于/大于等于
  public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
    List<Boolean> res = new ArrayList<Boolean>();
    int max = 0 ;
    for(int j = 0; j < candies.length; j++){
      if(candies[j] > max){
        max = candies[j];
      }
    }
    for(int i = 0; i < candies.length; i++){
      if(candies[i] + extraCandies >= max){
        res.add(true);
      } else {
        res.add(false);
      }
    }

    return res;

  }


  /***1512. 好数对的数目**
   * 注意边界条件，外层循环不应该包括最后的数字
   * 其他思路1：如果一个数出现了n次，那么这个数的好数对就是n*(n-1)/2
   * 其他思路2：1 <= nums.length <= 100
   *          1 <= nums[i] <= 100
   *由于有以上提示，说明数组的长度不超过100， 且数组的内的任何元素小于100，可以用一个长度101 的数组temp来存放nums 中每个元素出现的次数
   * nums 的数据当作temp的下标,第一次出现，+0，第二次出现+1，第三次出现+2
   *
   * **/
  public int numIdenticalPairs(int[] nums) {
    int res = 0;
    for(int i = 0; i<nums.length-1; i++){
      for(int j = i+1; j<nums.length;j++){
        if(nums[i] == nums[j]){
          res++;
        }
      }
    }
    return res;

  }
  public int numIdenticalPairs1(int[] nums) {
    int res = 0;
    Map<Integer, Integer> m = new HashMap<Integer, Integer>();
    for(int num: nums){
      m.put(num, m.getOrDefault(num, 0)+1);
    }
    for(Map.Entry<Integer,Integer> entry : m.entrySet()){
      int v =  entry.getValue();
      res += v * (v -1 ) /2;
    }

    return res;

  }
  //思路同上，但是用的是一个数组的下标和value来存储，而不是map
  public int numIdenticalPairs2(int[] nums) {
    int res = 0;
    int[] cnt = new int[101];
    for(int num:nums){
      cnt[num]++;
    }
    for(int i: cnt){
      if(i == 0){
        continue;
      } else {
        res += (i*(i-1))/2;
      }
    }
    return res;
  }

  //思路清奇,
  public int numIdenticalPairs3(int[] nums) {
    int res = 0;
    int[] temp = new int[101];
    for(int num : nums){
//      res += temp[num];
//      temp[num]++;
      res += temp[num]++;

    }
    return res;
  }

  /***1486. 数组异或操作
   *异或操作：递归, 对所有的元素进行操作时，就要考虑使用递归
   * ***/
  //为啥要从n-1开始：因为n是数组的长度，所以下标是0～n-1
  public int xorOperation(int n, int start) {
    if(n == 1){
      return start;
    }
    return start + 2 * (n-1) ^ xorOperation(n - 1, start);
  }
  public int xorOperation1(int n, int start) {
    int sum = 0;
    int[] nums = new int[n];
    for(int i = 0; i<nums.length; i++){
      nums[i] = start+ 2*i;
      sum = sum ^ nums[i];
    }
    return sum;

  }

  /***1395. 统计作战单位数
   * 其余思路：化简复杂度
   * 利用动态规划的方法 TODO
   ***/
  //暴力解决O(n^3)
  public int numTeams(int[] rating) {
    int res = 0;
    for(int i = 0; i < rating.length - 2; i++){
      for(int j = i+1; j < rating.length - 1; j++){
        for(int k = j+1; k < rating.length; k++){
          if((rating[i]<rating[j]&&rating[j]<rating[k]) ||(rating[i]>rating[j]&&rating[j]>rating[k])){
            res++;
          }
        }
      }
    }
    return res;
  }

  //复杂度O(n^2),枚举中间的j
  public int numTeams1(int[] rating) {
    int res = 0;
    if(rating.length < 3){
      return 0;
    }
    for(int j = 0; j<rating.length -1; j++){
      int iLess = 0;
      int iMore = 0;
      int kLess = 0;
      int kMore = 0;
      for(int i = 0; i<j; i++){
        if(rating[i]<rating[j]){
          iLess++;
        } else if(rating[i]>rating[j]){
          iMore++;
        }

      }
      for(int k = j+1; k<rating.length; k++){
        if(rating[k]<rating[j]){
          kLess++;
        } else if(rating[k]>rating[j]){
          kMore++;
        }
      }
      res += iLess*kMore + iMore*kLess;
    }
    return res;
  }

  //todo 需要写下思路
  public int numTeams2(int[] rating) {
    int res = 0;
    int n = rating.length;
    int[] min2Max = new int[n];
    int[] max2Min = new int[n];
    for(int i = 0; i<n; i++){
      for(int j = i-1; j>=0; j--){
        if(rating[j]>rating[i]){
          res+=max2Min[j];
          max2Min[i]++;
        }
        if(rating[j]<rating[i]){
          res+= min2Max[j];
          min2Max[i]++;

        }
      }
    }
    return res;
  }

  /***1313. 解压缩编码列表
   * ****/
  //太慢了
  public int[] decompressRLElist(int[] nums) {
    int len = 0;
    for(int i = 0; i<nums.length/2; i++){
      len += nums[2*i];
    }
    int[] res = new int[len];
    int index = 0;
    for(int i = 0; i< nums.length/2; i++){
      int cnt = nums[2*i];
      int val = nums[(2*i)+1];
      for(int j = 0; j<cnt; j++){
        res[index + j] = val;
      }
      index += cnt;
    }
    return res;

  }
  public int[] decompressRLElist2(int[] nums) {
    List<Integer> l = new ArrayList<Integer>() ;
    for(int i = 0; i<nums.length; i+=2){
      for(int j = 0; j<nums[i]; j++){
        l.add(nums[i+1]);
      }
    }
    return l.stream().mapToInt(num ->num).toArray(); //TODO arraylist 的流式写法

  }

  /***1365. 有多少小于当前数字的数字
   *
   * ***/
  //暴力解决
  public int[] smallerNumbersThanCurrent(int[] nums) {
    int[] res = new int[nums.length];
    for(int i = 0; i<nums.length; i++){
      int k = 0;
      for(int j = 0; j<nums.length; j++){
        if(nums[j] < nums[i]){
          k++;
        }
      }
      res[i]=k;
    }
    return res;

  }

  //todo 用复杂度更小的方法
  public int[] smallerNumbersThanCurrent1(int[] nums) {
    int[] res = new int[nums.length];

    return res;

  }

  /***
   * 1389. 按既定顺序创建目标数组*
   *  s1: 用链表
   *  s2：判断是直接添加数据，还是要在数组中插入数据；要插入数据时，进行平移操作；由于是向右平移，所以for循环要从后往前进行
   * */

  public int[] createTargetArray(int[] nums, int[] index) {
    List<Integer> l = new ArrayList<>();
    int len = nums.length;
    for(int i = 0; i < len; i++){
      l.add(index[i], nums[i]);
    }

    return l.stream().mapToInt(Integer::intValue).toArray();

  }

  public int[] createTargetArray1(int[] nums, int[] index) {
    int len = nums.length;
    int[] res = new int[nums.length];
    for(int i = 0; i < len; i++){
      if(index[i] >= i){
        res[index[i]] = nums[i];
      } else {
        for(int j = i - 1; j >=index[i]; j--){
          res[j+1]= res[j];
        }
        res[index[i]] = nums[i];
      }


    }
    return res;

  }

  /***1266. 访问所有点的最小时间***/
  public int minTimeToVisitAllPoints(int[][] points) {
    int res = 0;
    for(int i = 0; i < points.length-1; i++ ){
      int x1 = points[i][0];
      int y1 = points[i][1];
      int x2 = points[i+1][0];
      int y2 = points[i+1][1];
      res += Math.max(Math.abs(x2-x1), Math.abs(y2-y1));
    }
    return res;
  }

  /***面试题 08.04. 幂集**
   * for循环中，第二个用于判断的语句中的不应该有循环中的变量
   * */
  public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>(1 << nums.length);
    res.add(new ArrayList<>());
    for(int num: nums){
      for(int i = 0, j=res.size(); i<j; i++){
        List<Integer> temp = new ArrayList<>(res.get(i));
        temp.add(num);
        res.add(temp);
      }

    }
    return res;

  }

  /****1295. 统计位数为偶数的数字
   * 巧用数学知识，利用log函数
   * 注意double 转int
   * *****/
  public int findNumbers(int[] nums) {
    int res = 0;
    for(int i = 0; i<nums.length; i++){
     double d = Math.log10(nums[i]);
     int a = new Double(d).intValue();
     if ((a +1)%2 == 0) {
       res += 1;
     }
    }
    return res;

  }

  /*****1572. 矩阵对角线元素的和
   * ****/
  public int diagonalSum(int[][] mat) {
    int sum1 = 0;
    int sum2 = 0;
    for(int i = 0; i < mat.length; i++){
      sum1 += mat[i][i];
    }
    for(int i = 0, j = mat[0].length-1; i<=mat.length&&j>=0; i++,j--){
      if(i != j){
        sum2 += mat[i][j];
      }
    }

    return sum1+sum2;

  }

  /***1588. 所有奇数长度子数组的和
   * ***/
  public int sumOddLengthSubarrays(int[] arr) {
    int res = 0;
    for(int w = 1; w <= arr.length; w+=2){
      for(int l = 0, r = l + w; r <=arr.length; l++, r++){
        for(int i = l; i < r; i++){
          res += arr[i];
        }
      }
    }
    return res;

  }

  /*****1409. 查询带键的排列
   *
   * ****/
  public int[] processQueries(int[] queries, int m) {
    int[] res = new int[queries.length];
    ArrayList<Integer> l = new ArrayList<>();
    for(int i = 1; i <= m; i++){
      l.add(i);
    }
    for(int i = 0; i < queries.length; i++){
      int temp = l.indexOf(queries[i]);
      res[i] = temp;
      l.remove(temp);
      l.add(0,queries[i]);
    }

    return res;
  }

  /***1450. 在既定时间做作业的学生人数
   * ****/
  public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
    int res = 0;
    for(int i = 0; i < startTime.length; i++){
      if(startTime[i]<= queryTime && queryTime<= endTime[i]){
        res++;
      }
    }
    return res;
  }

  /***1534. 统计好三元组
   * 有多重for循环和多个if判断时，为了减少时间消耗，可以先进行判断，再进行for循环
   * ***/
  public int countGoodTriplets(int[] arr, int a, int b, int c) {
    int res = 0;
    for(int i = 0; i<arr.length; i++){
      for(int j = i +1; j <arr.length; j++){
        if(Math.abs(arr[i] - arr[j]) <= a){
          for(int k = j + 1; k<arr.length; k++){
            if(Math.abs(arr[j]- arr[k])<=b && Math.abs(arr[i]-arr[k])<= c){
              res++;
            }
          }
        }

      }
    }
    return res;

  }

  /*****59. 螺旋矩阵 II
   * 确定四个边界后，四个方向上依次循环
   * ******/
  public int[][] generateMatrix(int n) {
    int[][] res = new int[n][n];
    int l = 0;
    int r = n-1;
    int t = 0;
    int b = n-1;
    int num = 1;
    int tar = n*n;
    while(num <= tar){
      for(int i = l; i <= r; i++){
        res[t][i] = num;
        num++;
      }
      t++;

      for(int i = t; i<=b; i++){
        res[i][r] = num;
        num++;
      }
      r--;

      for(int i = r; i>=l; i--){
        res[b][i] = num;
        num++;
      }
      b--;

      for(int i = b; i>=t; i--){
        res[i][l] = num;
        num++;
      }
      l++;

    }
    return res;


  }

  /****1299. 将每个元素替换为右侧最大元素
   * 从右往左寻找最大值, 逆序遍历
   * ****/
  public int[] replaceElements(int[] arr) {
    int[] res = new int[arr.length];
    for(int i = 0; i< arr.length - 1; i++){
      int max = 0;
      for(int j = i+1; j<arr.length; j++){
        max = Math.max(arr[j], max);
      }
      res[i] = max;
    }
    res[arr.length - 1] = -1;
    return res;
  }

  public int[] replaceElements1(int[] arr) {
    int[] res = new int[arr.length];
    int max = -1;
    for(int i=arr.length-1; i>=0; i--){
      res[i] = max;
      if(arr[i] > max){
        max = arr[i];
      }

    }

    return res;
  }

  /****1329. 将矩阵按对角线排序
   * 冒泡排序,注意for循环的条件
   * todo 理清思路
   * ****/
  public int[][] diagonalSort(int[][] mat) {
    int m = mat.length;
    int n = mat[0].length;
    for(int l = 0; l < m - 1; l++){
      for(int i = 0; i < m - 1 - l; i++){
        for(int j = 0; j < n -1-l; j++){
          if(mat[i][j] > mat[i+1][j+1]){
            int temp = mat[i][j];
            mat[i][j] = mat[i+1][j+1];
            mat[i+1][j+1] = temp;

          }

        }
      }

    }
    return  mat;

  }

  /****950. 按递增顺序显示卡牌
   * todo
   * ***/
  public int[] deckRevealedIncreasing(int[] deck) {
    int[] res = new int[deck.length];


    return res;

  }

  /***1464. 数组中两元素的最大乘积
   * 两次遍历比较慢
   * ***/
  public int maxProduct(int[] nums) {
    int res = 0;
    for(int i = 0; i < nums.length; i++){
      for(int j = i+1; j < nums.length; j++){
        if(i != j){
          int temp = (nums[i]-1)*(nums[j]-1);
          res = Math.max(temp, res);
        }
      }
    }
    return res;
  }
  public int maxProduct1(int[] nums) {
    int res = 0;
    int len = nums.length;
    Arrays.sort(nums);
    res = (nums[len-1]-1) *(nums[len-2]-1);
    return res;
  }

  /****面试题 01.07. 旋转矩阵
   * 注意变换的对应关系
   * ****/
  public void rotate(int[][] matrix) {
    int[][] res = new int[matrix.length][matrix[0].length];
    for(int i = 0; i < matrix.length; i++){
      for(int j = 0; j < matrix[0].length; j++){
        res[j][matrix.length - 1 -i] = matrix[i][j];
      }
    }

    for(int i = 0; i < matrix.length; i++){
      for(int j = 0; j < matrix[0].length; j++){
        matrix[i][j] = res[i][j];
      }
    }

  }

  /****1351. 统计有序矩阵中的负数
   *强调是非递增序列，
   * 思路1：假设是递减的序列，那可以用二分法,二分法注意何时break
   * *****/

  public int countNegatives(int[][] grid) {
    int res = 0;
    int m = grid.length;
    int n = grid[0].length;
    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        if(grid[i][j] < 0){
          res ++;
        }
      }
    }
    return res;

  }

  public int countNegatives1(int[][] grid) {
    int res = 0;
    int m = grid.length;
    int n = grid[0].length;
    for(int i = 0; i < m; i++){
      int l = 0;
      int r = n - 1;
      while(l < r){
        int mid = l + (r-l)/2;
        if(grid[i][mid] < 0){
          if(mid == 0){
            res += n;
            break;
          }
          if(grid[i][mid - 1] >= 0){
            res += r - mid + 1;
            break;
          } else {
            r = mid - 1;
          }

        } else {
          l = mid + 1;
        }


      }

    }
    return res;

  }

  /***1252. 奇数值单元格的数目
   *  todo
   * ***/
  public int oddCells(int n, int m, int[][] indices) {
    int res = 0;
    return res;

  }

  /****832. 翻转图像
   * 翻转后，可以直接取反，不用再进行循环了
   * ****/
  public int[][] flipAndInvertImage(int[][] A) {
    int m = A.length;
    int n = A[0].length;
    int[][] res = new int[m][n];
    for(int i = 0; i<m; i++){
      for(int j = 0; j<n; j++){
        res[i][j] = A[i][n - 1- j] ^1;
      }

    }
    return res;

  }


  /****289. 生命游戏
   * 考虑数组的边界
   * *****/
  public void gameOfLife(int[][] board) {
    int[][] temp = new int[board.length][board[0].length];
    int m = board.length;
    int n = board[0].length;
    for(int i = 0; i<m; i++){
      for(int j = 0; j<n; j++){
        int cnt = 0;
        if(j < n-1 && board[i][j+1] == 1){
          cnt++;
        }

        if(i > 0 && j <n-1 && board[i-1][j+1] == 1){
          cnt++;
        }

        if(i <  m-1 && j < n-1 && board[i+1][j+1] == 1){
          cnt++;
        }

        if(j>0 && board[i][j-1] == 1){
          cnt++;
        }
        if(i > 0 && j>0 && board[i-1][j-1] == 1){
          cnt++;
        }
        if(i< m-1 && j>0 && board[i+1][j-1] == 1){
          cnt++;
        }
        if(i > 0 && board[i-1][j] == 1){
          cnt++;
        }
        if( i < m-1 && board[i+1][j] == 1){
          cnt++;
        }


        if(board[i][j] > 0){
          if(cnt < 2 || cnt > 3){
            temp[i][j] = 0;
          } else{
            temp[i][j] = 1;
          }
        } else {
          if(cnt == 3){
            temp[i][j] = 1;
          }

        }

      }
    }

    for(int i = 0; i<board.length; i++) {
      for (int j = 0; j < board[0].length; j++) {
        board[i][j] = temp[i][j];

      }
    }


  }

  /****1002. 查找常用字符
   * 暴力法较慢, 注意利用中间变量进行更新迭代
   * ******/
  public List<String> commonChars(String[] A) {
    List<String> res = new ArrayList<>();
    HashMap<String, Integer> map0 = new HashMap<>(); // 第一个字符串中的数据
    for(int j = 0; j< A[0].length(); j++){
      String s = String.valueOf(A[0].charAt(j));
      if(map0.containsKey(s)){
        int v = map0.get(s) + 1;
        map0.put(s, v);
      } else {
        map0.put(s, 1);
      }
    }

    Map<String, Integer> mapx = new HashMap<>(); //存放一个字符串的 数据
    Map<String, Integer> mapy = new HashMap<>(map0); // 复制一份map0，mapx与之进行对比进行更新，并重新赋给map0

    for(int i = 1; i<A.length; i++){
      String ss = A[i];

      for(int j = 0; j< A[i].length(); j++){
        String s = String.valueOf(A[i].charAt(j));
        if(mapx.containsKey(s)){
          int v = mapx.get(s) + 1;
          mapx.put(s, v);
        } else {
          mapx.put(s, 1);
        }
      }

      map0.forEach((k, v) -> {
        if(mapx.containsKey(k)){
          int count = mapx.get(k);
          if(v > count){
            mapy.put(k, count);
          }
        } else {
          mapy.remove(k);
        }
      });

      mapx.clear();
      map0 = new HashMap<>(mapy);

    }

    mapy.forEach((k,v) -> {
      if(v > 1){
        for(int i = 0; i < v; i++){
          res.add(k);
        }
      } else {
        res.add(k);
      }
    });

   return res;

  }

  // 生成一个长度为26的数组，数组的下标i表示26个字母
  public List<String> commonChars1(String[] A){
    int[] charCount = new int[26];
    List<String> res = new ArrayList<>();

    for(int i = 0; i < A[0].length(); i++){
      charCount[A[0].charAt(i) - 'a'] ++;
    }


    for(int i = 1; i<A.length; i++){
      int[] count = new int[26];
      for(int j = 0; j < A[i].length(); j++){
        count[A[i].charAt(j) - 'a']++;
      }

      for(int k = 0; k<26; k++){
        charCount[k] = Math.min(charCount[k], count[k]);
      }
    }

    for(int i = 0; i<26; i++){
      int c = charCount[i];
      for(int j = 1; j < c; j++){
        res.add(String.valueOf((char) (i+ 'a')));
      }

    }

    return res;
  }

  /****1460. 通过翻转子数组使两个数组相等
   * 只要两个数组内的所有元素都是相同的，且出现的次数也是相同的，就一定能够通过翻转实现
   * 所以只需要对两个数组进行排序，排成相同的顺序，再来对比两个数组是否相同就好
   * *****/
  public boolean canBeEqual(int[] target, int[] arr) {

    Arrays.sort(target);
    Arrays.sort(arr);

    return Arrays.equals(target, arr);

  }

  /*****1051. 高度检查器
   * 其实就是排序后和排序前的数组，有几个索引值的位置不同
   * ****/
  public int heightChecker(int[] heights) {
    int res = 0;

    int[] temp = Arrays.copyOf(heights,heights.length);
    Arrays.sort(heights);

    for(int i = 0; i<temp.length; i++){
      if(temp[i] != heights[i]){
        res ++;
      }

    }


    return res;

  }

  /*****1640. 能否连接形成数组
   * 由于整数各不相同，且不能调换顺序
   * 双重循时i++,则访问arr中的数据时，刚好是以array 的长度为间隔
   * **********
   * ****/
  public boolean canFormArray(int[] arr, int[][] pieces) {
   HashMap<Integer, int[]> map = new HashMap<>();

   for(int[] p : pieces){
     map.put(p[0],p);
   }

   for(int i = 0; i<arr.length;                ){
     if(!map.containsKey(arr[i])){
       return false;
     }
     int[] array = map.get(arr[i]);
     for(int j = 0; j < array.length; j++, i++){
       if(arr[i] != array[j]){
         return false;
       }

     }

   }
    return true;
  }


  /*****216. 组合总和 III
   * 回溯算法：类似枚举的搜索尝试过程，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径
   * 回溯通常用到递归的方法
   * todo 对回溯和递归的理解不够深入
   * ******/
  public List<List<Integer>> combinationSum3(int k, int n) {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> temp = new ArrayList<>();
    backtrack(res, temp, k, n, 1);
    return res;

  }

  private void  backtrack(List<List<Integer>> tar, List<Integer> list, int k, int n, int start){
    if(k > n||n < 0){
      return;
    }
    if(k == 0 && n == 0){
      tar.add(new ArrayList<>(list));
      return;
    }
    if(k > 0){ //剪枝
      for(int i = start; i<=9; i++){
        list.add(i);
        backtrack(tar, list, k-1, n-i, i+1);
        list.remove(list.size()- 1); //撤销的操作，只有回溯中会用到撤销
      }
    }
  }


  /****977. 有序数组的平方
   * 双指针方法比较快
   * *****/
  public int[] sortedSquares(int[] nums) {
    int[] res = new int[nums.length];
    for(int i = 0; i < nums.length; i++){
      res[i] = nums[i] * nums[i];

    }
    Arrays.sort(res);
    return res;

  }
  public int[] sortedSquares1(int[] nums) {
    int n = nums.length;
    int[] res = new int[n];
    for(int i = 0, j = n - 1, pos = n-1; i<=j;){
      if(nums[i] * nums[i] < nums[j] * nums[j]){
        res[pos] = nums[j] * nums[j];
        j--;
      } else {
        res[pos] = nums[i] * nums[i];
        i++;
      }
      pos--;
    }

    return res;

  }

  /*****1502. 判断能否形成等差数列
   * ******/
  public boolean canMakeArithmeticProgression(int[] arr) {
    Arrays.sort(arr);
    for(int i = 0; i<arr.length - 2; i++){
      if(arr[i+1] - arr[i] !=  arr[i+2] - arr[i+1]){
        return false;
      }
    }
    return true;

  }

  /****561. 数组拆分 I
   * ****/
  public int arrayPairSum(int[] nums) {
    int res = 0;
    Arrays.sort(nums);
    for(int i = 0; i<nums.length; i+=2){
      res += nums[i];
    }
    return res;

  }

  /****1380. 矩阵中的幸运数
   * ****/
  public List<Integer> luckyNumbers (int[][] matrix) {
    List<Integer> res = new ArrayList<>();
    int m = matrix.length;
    int n = matrix[0].length;
    for(int i = 0; i<m; i++) {
      int tempMin = Integer.MAX_VALUE;
      int x = 0;
      int y = 0;
      for (int j = 0; j < n; j++) {
        if (matrix[i][j] < tempMin) {
          tempMin = matrix[i][j];
          x = i;
          y = j;
        }
      }
      if(judge(x,y,matrix)){
        res.add(tempMin);
      }


    }
    return res;
  }

  public static boolean judge(int x,int y,int[][] matrix) {
    for(int i = 0 ; i < matrix.length ; i++) {
      if(matrix[x][y] < matrix[i][y]) {
        return false;
      }
    }
    return true;
  }

  public static void main(String[] args) {
    ArrayTest arrayTest = new ArrayTest();
    System.out.println("Hello World!");
    int[] tar = {1,2, 3,4};
    int[] arr = {2,1, 4,3};
    int[][] a = {{36376,85652,21002,4510},{68246,64237,42962,9974},{32768,97721,47338,5841},{55103,18179,79062,46542}};
    List<Integer> r = arrayTest.luckyNumbers(a);


  // boolean c = arrayTest.canBeEqual(tar, arr);
    System.out.println(r.size());
  }


}
