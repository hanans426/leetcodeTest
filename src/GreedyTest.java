import java.security.acl.Group;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.ArrayList;

/**
 * User: gaohan
 * Date: 2020/12/28
 * Time: 14:45
 * 贪心算法的思想
 */
public class GreedyTest {

  /*****455. 分发饼干
   * *******/
  public int findContentChildren(int[] g, int[] s) {
    Arrays.sort(g);
    Arrays.sort(s);
    int i = 0; int j = 0;
    while(i < g.length  && j <s.length){
      if(g[i] <= s[j]){
        i++;
      }
      j++;
    }
    return i;
  }


  /*****435. 无重叠区间
   * 移除区间的最小数量，等同于不重叠区间的最大数量
   * *****/
  public int eraseOverlapIntervals(int[][] intervals) {
    if(intervals.length == 0){
      return 0;
    }
    int len = intervals.length;
    int cnt = 1;
    Arrays.sort(intervals, Comparator.comparingInt(o->o[1])); //将其按照尾部元素进行排序,
    int end = intervals[0][1];
    for(int i = 0; i<len; i++){
      if(intervals[i][0] < end){
        continue; //结束本次循环，跳过接下来循环体中的一些语句，进行最内层循环中的下一次循环
      }
      end = intervals[i][1];
      cnt++;
    }

    return len - cnt;

  }
  /*****452. 用最少数量的箭引爆气球
   * 上一题的变形，也是求不重叠区间的个数，
   * ******/
  public int findMinArrowShots(int[][] points) {
    if(points.length == 0){
      return 0;
    }
    int len = points.length;
    int cnt = 1;
    Arrays.sort(points, Comparator.comparingInt(o->o[1]));
    int end = points[0][1];
    for(int i = 0; i< len; i++){
      if(points[i][0]<= end){
        continue;
      }
      end = points[i][1];
      cnt++;
    }
    return cnt;

  }

  /******406. 根据身高重建队列
   * 身高降序，k值升序
   * 身高较高的同学首先做插入操作，否则身高较小的同学原先正确的插入第K个位置的话，在插入较高的同学，那么较小同学的k值就不正确
   * ******/
  public int[][] reconstructQueue(int[][] people) {
    if(people.length == 0 || people[0].length == 0 || people == null){
      return new int[0][0];
    }
    Arrays.sort(people, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        if(o1[0] != o2[0]){
          return o2[0] - o1[0]; // 两人身高不同时，身高较大的排在前边
        } else {
          return o1[1] - o2[1]; //两人身高相同时，K值较大的排在前边
        }
      }
    });
//    int[][] result = new int[people.length][people[0].length];
//    for(int i = 0; i<people.length; i++){
//      if(people[i][1] >= i){
//        result[i] = people[i];
//      } else {
//        for(int j = i-1; j>=people[i][1]; j--){
//          result[j+1] = result[j];
//        }
//        result[people[i][1]] = people[i];
//      }
//    }
//    return result;
    List<int[]> queue = new ArrayList<>();
    for (int[] p : people) {
      queue.add(p[1], p);
    }
    return queue.toArray(new int[queue.size()][]);

  }

  /*****763. 划分字母区间
   * *****/
//  public List<Integer> partitionLabels(String S) {
//
//  }


  /*****605. 种花问题
   * 两个花朵之间至少存在一块空地
   * *******/
  public boolean canPlaceFlowers(int[] flowerbed, int n) {
       int len = flowerbed.length;
    int cnt = 0;
    for(int i = 0; i<len&&cnt<n; i++){
      if(flowerbed[i] == 1){
        continue;
      }
      int pre = i == 0? 0:flowerbed[i-1];
      int next = i==len-1?0:flowerbed[i+1];

      if(pre == 0 && next == 0){
        cnt ++;
        flowerbed[i] = 1;
      }
    }

    return cnt>=n;  //如果 种完了，则说明是可以的，没有种完则不可以

  }

  /*****392. 判断子序列
   * *****/
  public boolean isSubsequence(String s, String t) {
    int index = -1;
    for(char c: s.toCharArray()){
      index = t.indexOf(c,index +1);
      if(index == -1){
        return false;
      }
    }
    return true;
  }

  //双指针
  public boolean isSubsequence1(String s, String t) {
   int l1 = s.length();
   int l2 = t.length();
   int i = 0, j= 0;
   while(i < l1 && j < l2){
     if(s.charAt(i) == t.charAt(j)){
       i++;
     }
     j++;
   }
   return i==l1;

  }

  /*****665. 非递减数列
   * nums[i] <= nums[i + 1]
   * 注意点：不能只注意cnt, 还要对数组进行修改
   * *******/
  public boolean checkPossibility(int[] nums) {
    if(nums == null || nums.length ==0){
      return false;
    }
    int cnt = 0;
    for(int i = 1; i < nums.length && cnt <= 1; i++) {

      if (nums[i-1] > nums[i]) {
        cnt++;

        //为何这么做呢？
        //对数组进行修改，看具体修改哪个位置上的数据，如果i-2位的数据较大，则修改i位的数据，反之修改i-1位对的数据
        if(i-2 >= 0 && nums[i-2] > nums[i]){
          nums[i] = nums[i-1];
        } else {
          nums[i-1] = nums[i];
        }
      }
    }

    return cnt <=1;

  }

  /******122. 买卖股票的最佳时机 II
   * 在局部最优的情况下保证全局最优
   * *******/
  public int maxProfit(int[] prices) {
    int res = 0;
    for(int i = 1; i<prices.length; i++){
      if(prices[i] > prices[i-1]){
        res += (prices[i] - prices[i-1]);
      }
    }
    return res;

  }

  public static void main(String[] args){
    System.out.println("hello greedy");
    GreedyTest greedyTest = new GreedyTest();
    int[] a = {3,4,2,3};
    boolean b =  greedyTest.checkPossibility(a);

  }
}


