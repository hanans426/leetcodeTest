import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
/**
 * User: gaohan
 * Date: 2020/12/25
 * Time: 16:21
 *
 * 双指针算法的思想
 */
public class DoublePointTest {

  /****167. 两数之和 II - 输入有序数组
   * 注意是有序数组，for循环中的限制条件是i<j， 修改条件要根据具体情况进行调整
   * ****/
  public int[] twoSum(int[] numbers, int target) {
    int[] res = new int[2];
    for(int i = 0, j = numbers.length-1;  i<j;){
      if(numbers[i] + numbers[j] == target) {
        res[0] = i + 1;
        res[1] = j + 1;
      } else if(numbers[i] + numbers[j] < target){
        i++;
      } else {
        j--;
      }
    }
    return res;
  }

  /*******633. 平方数之和
   * ********/
  public boolean judgeSquareSum(int c) {
    for(int i = 0, j =(int) Math.sqrt(c); i <= j;){
      if(i*i + j*j == c){
        return true;
      } else if(i*i + j*j > c){
        j--;
      } else {
        i++;
      }
    }
    return false;
  }

  /*****编写一个函数，以字符串作为输入，反转该字符串中的元音字母。
   * for 循环的最后一个语句，
   * ******/

  public String reverseVowels(String s) {

    final HashSet<Character> vowels = new HashSet<>(
      Arrays.asList('a','e', 'i','o', 'u','A','E','I', 'O', 'U')
    );
    char[] res = new char[s.length()];

    for(int i = 0, j = s.length() - 1; i<j;){
      char ci = s.charAt(i);
      char cj = s.charAt(j);
      if(!vowels.contains(ci)){
        res[i] = ci;
        i++;
      } else if(!vowels.contains(cj)){
        res[j] = cj;
        j--;
      } else {
        res[i] = cj;
        res[j] = ci;
        i++;
        j--;
      }
    }
    return new String(res);

  }

  /****680. 验证回文字符串 Ⅱ
   * 循环体中，一定要对边界条件进行改变
   * ****/
  public boolean validPalindrome(String s) {
    int i = 0;
    int j = s.length()-1;
    while(i < j){
      if(s.charAt(i) != s.charAt(j)){
        return isPalindrome(s, i+1, j) || isPalindrome(s, i, j-1);
      }
      i++;
      j--;

    }
    return true;

  }

  private boolean isPalindrome(String s, int i, int j){
    while(i < j){
      if(s.charAt(i) != s.charAt(j)){
        return false;
      }
      i++;
      j--;
    }
    return true;
  }

  /*****88. 合并两个有序数组
   * todo  归并排序
   * ****/
  public void merge(int[] nums1, int m, int[] nums2, int n) {
    int index = m+n-1;
    int index1 = m-1;
    int index2 = n-1;
    while(index1 >=0 && index2 >=0){
      nums1[index--] = nums1[index1]>nums2[index2]? nums1[index1--]:nums2[index2--];

      System.arraycopy(nums2, 0, nums1, 0, index2 + 1);

    }

    int p1 = m - 1;
    int p2 = n - 1;
    int p = m + n - 1;

    while ((p1 >= 0) && (p2 >= 0))

      nums1[p--] = (nums1[p1] < nums2[p2]) ? nums2[p2--] : nums1[p1--];

    // add missing elements from nums2
    System.arraycopy(nums2, 0, nums1, 0, p2 + 1);

  }

  /********524. 通过删除字母匹配到字典里最长单词
   * ********/
  public String findLongestWord(String s, List<String> d) {
    String longestWord = "";
    for(String t:d){
      int l1 = longestWord.length();
      int l2 = t.length();
      if(l1 > l2 || (l1 == l2 && longestWord.compareTo(t) < 0)){
         continue;
      }
      if(isValid(s, t)){
        longestWord = t;
      }

    }
    return  longestWord;

  }

  private boolean isValid(String s, String target){
    int i = 0;
    int j = 0;
    while(i < s.length() && j < target.length()){
      if(s.charAt(i) == target.charAt(j)){
        j++;
      }
      i++;
    }
    return j==target.length();
  }


  public static void main(String[] args){
    System.out.println("Hello doublePointer");
  }
}
