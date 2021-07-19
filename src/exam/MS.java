package exam;

/**
 * User: gaohan
 * Date: 2021/3/22
 * Time: 15:35
 */
public class MS {

  public static int solution2(int[] arrs){
    if(arrs == null || arrs.length == 0){
      return -1;//不存在
    }
    int left = 0;
    int right = arrs.length - 1;

    while(left < right){
      int mid = left + (right - left)/2;
      if(arrs[mid] == mid){
        return mid;
      } else if(arrs[mid] > mid){
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return  -1;

  }
  static int res = -1;
  public static int solution(int[] arrs){
    if(arrs == null || arrs.length == 0){
      return -1;//不存在
    }

    help(arrs, 0, arrs.length-1);
    return res;
  }
  private static void help(int[] arrs, int l, int r){
    if(l > r){
      return;
    }

    int m = l + (r - l)/2;
    if(arrs[m] == m){
      res = m;
      return;
    }
    help(arrs, l, m-1);
    help(arrs, Math.max(arrs[m], m+1) , r); //优化
  }

  public static void main(String[] args) {
    int[] arr = {-1,0,4,4,4};
    int[] arr1 = {-1,1, 1, 3,3};
    int res = solution(arr1);
    System.out.println(res);
  }
}
