/**
 * User: gaohan
 * Date: 2021/8/11
 * Time: 11:38
 */
public class SubArray {
  /**
   * 300. 最长递增子序列
   */
//  public int lengthOfLIS(int[] nums) {
//
//  }

  /**
   *只有0和1的两个元素的一个数组，数组中0和1出现次数相同的最长递增子序列
   */
  public static int getMax(int[] arr){
    for(int i=0; i<arr.length; i++){
      if(arr[i] == 0){
        arr[i] = -1;
      }
    }
    int sum = 0;
    int max = 0;
    for(int i = 0;i<arr.length; i++){
      sum = arr[i];
      for(int j = i+1; j<arr.length; j++) {
        sum += arr[j];
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

  public static void main(String[] args) {
    int[] arr = new int[]{0,0,1,0,1,0};
    int res = getMax(arr);
    System.out.println(res);
  }


}
