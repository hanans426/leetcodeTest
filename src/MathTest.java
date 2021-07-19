import java.util.ArrayList;
import java.util.Arrays;

/**
 * User: gaohan
 * Date: 2020/12/28
 * Time: 17:37
 * 数学类的算法
 * 概念：
 * - 素数：又称质数，除了1和本身之外，不能被其他任何自然数整除
 * - 素数分解：每一个数都可以分解为素数的乘积
 *
 */
public class MathTest {

  /******204. 计数质数
   *
   * *******/
  //暴力解法，超出时间限制
  public int countPrimes(int n) {
    int res = 1;
    if(n<=2){
      return 0;
    }
    for(int i = 3; i < n; i++){
      if(isPrimes(i)){
        res++;
      }
    }

    return res;

  }

  public boolean isPrimes(int n){
    for(int i = 2; i*i<= n; i++){
      if(n % i == 0){
        return false;
      }
    }
    return true;
  }

 //埃拉托斯特尼筛法在每次找到一个素数时，将能被素数整除的数排除掉。
  public int countPrimes1(int n) {
    boolean[] isPrimes = new boolean[n];
    Arrays.fill(isPrimes, true);
    for(int i = 2;i*i < n; i++){
      if(isPrimes[i]){
        for(int j = i*i; j< n; j+=i){
          isPrimes[j] = false;
        }
      }
    }
    int res = 0;

    for(int i = 2; i < n; i++){
      if(isPrimes[i]){
        res++;
      }
    }
    return res;

  }

  //求最大公约数
  public int gcd(int a, int b){
    return b==0? a:gcd(b, a%b);
  }

  //求最小公倍数：两数的乘积除以最大公约数
  public int lcm(int a, int b){
    return a*b/gcd(a,b);
  }


  /******使用位操作和减法求解最大公约数
   * ******/
  //进制转换：先取余，再整除取整
  /****504. 七进制数
   * ****/
  public String convertToBase7(int num) {
    if(num == 0){
      return "0";
    }
    boolean flag = num < 0;
    StringBuilder res = new StringBuilder();
    num = Math.abs(num);
    while(num != 0){
      res.append(num%7);
      num = num / 7;
    }

    if(flag){
      res.append("-");
    }
    return res.reverse().toString();

  }

  /*****405. 数字转换为十六进制数
   * 核心思想：使用位运算，每4位对应1位16进制数字
   * ******/
  public String toHex(int num) {
    if(num == 0){
      return "0";
    }
    char[] hex = "0123456789abcdef".toCharArray();
    StringBuilder sb = new StringBuilder();
    while (num != 0){
      int temp = num & 0xf; //取低4位的十进制，0xf = 0000000000001111,保留了后四位的数值
      sb.append(hex[temp]); //映射对应字符
      num >>>= 4;   //逻辑右移四位
    }
    return sb.reverse().toString();

  }

  /*****168. Excel表列名称
   * *****/
  public String convertToTitle(int n) {
    if(n<=0){
      return "";
    }
    StringBuilder sb = new StringBuilder();
    while(n > 0){
      n--; //因为没有0,所以要减1
     sb.append((char) (n % 26 + 'A'));
     n = n/26;
    }
    return sb.reverse().toString();

  }

  /******172. 阶乘后的零
   * ****/
  //没有给出n的范围，暴力解法容易溢出
  public int trailingZeroes(int n) {
    int res = 1;
    int cnt = 0;
    for(int i = 1; i<=n; i++){
      res = res * i;
    }
    String s = "" + res;
    for(int i = s.length()-1; i>0;){
      if(s.charAt(i) == '0'){
        cnt++;
        i--;
      } else{
        return cnt;
      }
    }
    return cnt;
  }

  //只有2*5才会产生0，2的因子数很多，所以只需考虑乘数因子中有多少个5
  public int trailingZeroes1(int n) {
    int cnt = 0;
    while(n >= 5){
      cnt += n/5;
      n = n/5;
    }
    return cnt;
   // return n==0?0:n/5+trailingZeroes(n/5);

  }

  /******67. 二进制求和
   * *****/
  public String addBinary(String a, String b) {
    int i = a.length() - 1;
    int j = b.length() - 1;
    StringBuilder sb = new StringBuilder();
    int temp = 0;
    while(i >= 0 || j>= 0 || temp == 1){
      int x = i>=0 ? a.charAt(i) - '0':0;
      int y = j>=0?  b.charAt(j) - '0':0; //一定要判断，否则会越界
      if(x+y+temp == 0){
        sb.append("0");
        temp =0;
      } else if(x+y+temp == 1){
        sb.append("1");
        temp = 0;
      } else if(x+y+temp == 2){
        sb.append("0");
        temp =1;
      } else if(x+y+temp == 3){
        sb.append("1");
        temp =1;
      }
      i--;
      j--;
        //利用取余和取整来减少判断
//      if(i >= 0 && a.charAt(i)=='1'){
//        temp++;
//      }
//      if(j >=0 && b.charAt(j) == '1'){
//        temp++;
//      }
//      sb.append(temp%2);
//      temp = temp/2;
//      i--;
//      j--;
    }
    return sb.reverse().toString();

  }
  /*****415. 字符串相加
   * *****/
  public String addStrings(String num1, String num2) {
    int i = num1.length()-1;
    int j = num2.length()-1;
    int temp = 0;
    StringBuilder sb = new StringBuilder();
    while(i>=0||j>=0||temp==1){
      int x = i >= 0?num1.charAt(i) -'0':0;
      int y = j >=0?num2.charAt(j) - '0':0;
      sb.append((x+y+temp)%10);
      temp = (x+y+temp)/10;
      i--;
      j--;
    }
    return sb.reverse().toString();
  }

  /******462. 最少移动次数使数组元素相等 II
   * 全部变成中位数是最优解
   * *******/
  public int minMoves2(int[] nums) {
    Arrays.sort(nums);
    int l = 0;
    int r = nums.length-1;
    int res = 0;
    while( l<r){
      res += nums[r]-nums[l];
    }

    return res;

  }


  /*****169. 多数元素
   * ******/
  public int majorityElement(int[] nums) {
    Arrays.sort(nums);
    return nums[nums.length/2];
  }

  /****367. 有效的完全平方数
   * 用遍历的方法会超出时间限制
   * ****/
  public boolean isPerfectSquare(int num) {
    for(int i = 1; i<=num; i++){
      if(i*i == num){
        return true;
      }
    }
    return false;
  }

  //二分法
  public boolean isPerfectSquare1(int num) {
     int low = 1;
     int high = num;
     while (low <= high){
       int mid = low + (high-low)/2;
       int n = num / mid;
       if(n == mid){
         if(num % mid == 0){
           return true;
         }
         low = mid+1;
       }
       if(n < mid){
         high = mid;
       }
       if(n > mid){
         low = mid+1;
       }
     }
     return false;
  }

  //数学解法：完全平方数之间的差值为2等差数列，
  public boolean isPerfectSquare2(int num) {
    int subnum = 1;
    while(num > 0){
      num -= subnum;
      subnum +=2;
    }
    return num==0;
  }

  /******326. 3的幂
   * 1162261467是3的19次幂，是整数范围内最大的3的幂次，如果能被最大幂次整除，一定是3的幂
   * *****/
  public boolean isPowerOfThree(int n) {
    return n > 0 && 1162261467%n == 0;
  }

  /*****628. 三个数的最大乘积
   * *****/
  public int maximumProduct(int[] nums) {
    Arrays.sort(nums);
    int l =  nums.length;
    int n = nums[l- 1]*nums[l-2]*nums[l-3];
    int m = nums[0]*nums[1]*nums[l-1];
    return Math.max(n,m);
  }

  /*****238. 除自身以外数组的乘积
   * 乘积数组，
   * 正循环一遍
   * 逆循环一遍
   * ********/
  public int[] productExceptSelf(int[] nums) {
     int[] res = new int[nums.length];
     int left = 1;
     int right = 1;
     for(int i = 0; i< nums.length; i++){
       res[i] = left;
       left *= nums[i];
     }
     for(int j = nums.length-1; j >= 0; j--){
       res[j] *=  right;
       right *= nums[j];
     }
     return res;

  }

  public static void main(String[] args){
    System.out.println("hello math");
    MathTest mathTest = new MathTest();
    int g = mathTest.gcd(6,12);
    int l = mathTest.lcm(6,12);
    int t = mathTest.trailingZeroes(13);
    String a = mathTest.addBinary("10","10");
    System.out.println(a);
  }
}
