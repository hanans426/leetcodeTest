/**
 * User: gaohan
 * Date: 2020/12/28
 * Time: 17:34
 * 二分法的难点是：边界条件的确定，
 * 如果h的赋值表达式为h=m,那么循环条件为l < h
 *
 */
import java.util.List;

public class PartitonTest {

  //二分查找的算法时间复杂度为O（logN）
  public int binarySearch(int[] nums, int key){
    int l = 0;
    int h = nums.length -1;
    while(l <= h){
      int m = l + (h-l)/2; //m 放在循环内部
      if(nums[m] == key){
        return m;
      } else if(nums[m] < key){
        l = m+1;
      } else {
        h = m-1;
      }
    }
    return -1;
  }

  public int binarySearch1(int[] nums, int key){
    int l = 0;
    int h = nums.length -1;
    while(l < h){
      int m = l + (h-l)/2;  //m放在循环内部
      if(nums[m] >= key){
        h = m;
      } else {
        l = m+1;
      }
    }
    return l;
  }

  /****69. x 的平方根
   * *******/
  public int mySqrt(int x) {
    if(x <= 1){
      return x;
    }
    int l = 0;
    int h = x;
    while(l <= h){ //注意边界条件，
      int m = l + (h-l)/2;
      int sqrt = x/m;
      if(sqrt == m){
        return m;
      } else if(sqrt < m){
        h = m -1;
      } else {
        l = m+1;
      }

    }
    return h;

  }

  /******744. 寻找比目标字母大的最小字母
   * *******/
  public char nextGreatestLetter(char[] letters, char target) {
    int l = 0;
    int h = letters.length - 1;
    while(l <= h){
      int m = l + (h-l)/2;
      if(letters[m] <= target){  // l=h时，刚好==target,要找的是比target大的最小字母，所以 了要增加一位
        l = m+1;
      } else {
        h = m-1;
      }
    }
    return l< letters.length -1?letters[l]:letters[0];
  }

  /******540. 有序数组中的单一元素
   *利用下标索引进行判断
   * 判断mid两边的数组长度是奇数还是偶数，
   * 选择是奇数的一方继续二分
   * // todo
   * ******/
  public int singleNonDuplicate(int[] nums) {
    int l = 0;
    int h = nums.length;
    while(l < h) {
      int m = l + (h - l) / 2;
      if(m % 2 == 1){ //下标是奇数时，表示是第偶数个数字,前边有奇数个数字

      }
      if(nums[m] == nums[m+1]){ //独立值在右侧区间
        l = m+2;
      } else {
        h = m; //独立值在左侧区间
      }

    }
    return nums[l];
  }



  /****278. 第一个错误的版本
   * ******/

  /******153. 寻找旋转排序数组中的最小值
   * *******/
  public int findMin(int[] nums) {
    int l = 0;
    int h = nums.length - 1;
    while(l < h){
      int m = l + (h-l)/2;
      if(nums[m] > nums[h]){
        l = m+1;
      } else {
        h = m;
      }
    }
    return nums[l];

  }

  /*****34. 在排序数组中查找元素的第一个和最后一个位置
   * *****/
  public int[] searchRange(int[] nums, int target) {
    int start = binarySearch2(nums, target);
    int end = binarySearch2(nums, target+1)-1;
    if(start == nums.length || nums[start] != target){
      return  new int[]{-1, -1};
    } else {
      return new int[]{start, Math.max(start,end)};
    }

  }

  private static int binarySearch2(int[] nums, int target){
    int l = 0;
    int h = nums.length ; //注意h的初始值，
    while(l < h){
      int m = l + (h-l)/2;
      if(nums[m] >= target){
        h = m;
      } else {
        l = m+1;
      }
    }
    return l;
  }

  /*****241. 为运算表达式设计优先级
   * 分治的思想
   * todo
   * *****/
//  public List<Integer> diffWaysToCompute(String input) {
//
//  }

  public static void main(String[] args){
    System.out.println("hell0 partition");
    PartitonTest partitonTest = new PartitonTest();
    int[] a = {1,2,3,3,4,5,6};
    int[] aa = {1,2,3,4,5};
    int b = partitonTest.binarySearch1(a, 3);
    System.out.println(b);
  }
}
