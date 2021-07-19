import java.util.Arrays;
import java.util.PriorityQueue;

/**
 * User: gaohan
 * Date: 2020/12/24
 * Time: 15:44
 * 排序算法
 */
public class Sort {

  /****215. 数组中的第K个最大元素
   * 堆排：大顶堆和小顶堆 的思想
   * 快排
   * ***/
  //数组的排序，时间复杂度O(NlogN),空间复杂度O(1)
  public int findKthLargest(int[] nums, int k) {
    Arrays.sort(nums);
    return nums[nums.length - k];
  }
//  // 快排
//  public int findKthLargest1(int[] nums, int k) {
//    int l = 0;
//    int r = nums.length - 1;
//    while (l < r){
//      if(nums[l] < )
//
//    }
//
//
//    return nums[nums.length - k];
//  }
//
//  private int partition(int[] nums, int l, int r){
//    int i = l+1;
//    int j = r + 1;
//    while (true){
//      if(nums[i] < n )
//    }
//
//
//  }
  /*****快排算法------------------------
   * 快排的递归算法
   * 选择左边的为主要元素时，要从右开始比较
   * *****/
  public int[] quickSort(int[] arr){
    partition(arr, 0, arr.length-1);
    return arr;

  }
  private void partition(int[] arr, int l, int r){
    if(l >= r){
      return; // 递归结束的条件
    }
    int left = l;
    int right = r;
    int pivot = arr[l]; //选择一个主要元素
    while (left < right){
      //首先从右往左寻找比主元素小的
      while(left < right && arr[right] >= pivot) right--; //  比pivot大的元素，则移动右指针
     // arr[left] = arr[right]; //找到之后把它放到左边

      while(left < right && arr[left] <= pivot) left++;
      //arr[right] = arr[left];

      swap(arr, left, right);
    }
    // 循环结束时，left = right
    //arr[left] = pivot;
    swap(arr, left, l);
    partition(arr, l, left-1);
    partition(arr, left+1, r);

  }
  private void swap(int[] arr, int i, int j){
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }

  //堆排 时间复杂度 O(NlogK)，空间复杂度 O(K)。
  // 先进先出，默认是最小堆，每次取出的都是最小值
  public int findKthLargest2(int[] nums, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    for (int val : nums){
      pq.add(val);
      if(pq.size() > k){
        pq.poll();
      }
    }
    return pq.peek();
  }

  /*****堆排序：
   * ****/
  public void heapSort(int[] arr){

  }

  public static void main(String[] args){
    System.out.println("Hello sort");
  }
}
