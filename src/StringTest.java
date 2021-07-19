import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Stack;

/**
 * User: gaohan
 * Date: 2020/12/22
 * Time: 10:46
 *
 */
public class StringTest {


  /****面试题 16.26. 计算器
   * ****/
  public int calculate(String s) {
    Stack<Integer> stack = new Stack<>();
    int num = 0;
    char operater = '+';

    for(int i = 0; i< s.length(); i++){
      char c = s.charAt(i);
      if(Character.isDigit(c)){
        int ci = c-'0';
        num = num*10 + ci;
      }

      if((!Character.isDigit(c) && c != ' ') ||i == s.length() - 1 ){ //注意算式的最后一位，数组要进行符号的判断
        switch (operater){
          case '+':
            stack.push(num);
            break;
          case '-':
            stack.push(-num);
            break;
          case '*':
            num = num * stack.pop();
            stack.push(num);
            break;
          case '/':
            num =  stack.pop() / num;
            stack.push(num);
            break;

        }
        num = 0; //此处更新num 的值
        operater = c;  //此处更新operater

      }

    }
    int res = 0;
    while (!stack.isEmpty())
      res += stack.pop();

    return res;

  }


  /****1576. 替换所有的问号
   * 只需返回一个可能的答案，所有只需对所有的？都替换成一个字符，如果附近有相同字符的就替换为下一个字符
   * ****/
  public String modifyString(String s) {
    char[] sb = s.toCharArray();
    for(int i = 0; i<s.length(); i++){
      char ch = s.charAt(i);
      if(ch == '?'){
        char temp_l = i==0? ' ': sb[i-1];
        char temp_r = i==s.length() - 1? ' ': sb[i+1];
        char a = 'a';

        while (a == temp_l || a == temp_r){
          a++;
        }

        sb[i] = a;
      }

    }

    return new String(sb);
  }

  /****459. 重复的子字符串
   * 如果没有循环字符串，那么两个s 组合后，从1开始找子字符串,一定是s.length处才能找到，如果存在重复字符，在之前就能找到
   * ***/
  public boolean repeatedSubstringPattern(String s) {
    return (s+s).indexOf(s, 1) != s.length();
  }

  //常规解法：KMP算法的经典问题
//  public boolean repeatedSubstringPattern1(String s) {
//
//  }


  /******1370. 上升下降字符串
   * 桶排序: 构建一个数组，下标表示26个字母，数值表示该字母出现的次数
   * ****/
  public String sortString(String s) {
    char[] ch = s.toCharArray();
    int[] bucket = new int[26];
    for(int i = 0; i< ch.length; i++){
      bucket[ch[i] - 'a']++;
    }
    StringBuffer res = new StringBuffer();
    while(res.length() < s.length()){ // 没有取完时
      for(int i = 0; i<bucket.length; i++){
        if(bucket[i] > 0){
          res.append((char)('a'+ i));
          bucket[i]--;
        }
      }

      for(int j = bucket.length -1; j>=0; j--){
        if(bucket[j] > 0){
          res.append((char)('a'+j));
          bucket[j]--;
        }
      }
    }
    return res.toString();

  }

  /***面试题 01.06. 字符串压缩
   * *****/
  // 理解错题意了
  public String compressString(String S) {
    int[] map = new int[26];
    char[] chars = S.toCharArray();
    for(int i = 0; i<chars.length; i++){
      map[chars[i]-'a']++;
    }
    StringBuffer res = new StringBuffer();
    for(int i = 0; i<map.length; i++){
      if(map[i] > 0){
        res.append((char)(i + 'a'));
        res.append(map[i]);
      }
    }
    return res.toString();
  }
 //模拟法
  public String compressString1(String S) {
    char[] chars = S.toCharArray();
    if(chars.length == 0){
      return S;
    }
    char c = chars[0];
    int cnt = 0;
    StringBuffer res = new StringBuffer();
    for(int i = 0; i<chars.length; i++){
      if(chars[i] == c){
        cnt++;
      } else {
        res.append(c);
        res.append(cnt);
        c = chars[i];
        cnt = 1;
      }
    }
    //这两行不要忘记，是最后的字符
    res.append(c);
    res.append(cnt);

    return  res.toString().length()>= S.length()? S: res.toString();


  }


  //快慢指针法
  public String compressString2(String S) {
    char[] chars = S.toCharArray();
    if(chars.length == 0){
      return S;
    }
    int slow = 0;
    int fast = 1;
    StringBuffer res = new StringBuffer();
    while(fast < chars.length){
      if(chars[slow] != chars[fast]){
        res.append(chars[slow]);
        res.append(fast - slow);
        slow = fast;
        fast++;
      } else {
        fast++;
      }
    }
    res.append(chars[slow]);
    res.append(fast - slow);
    return  res.toString().length()>= S.length()? S: res.toString();

  }


  /***牛客： 求出字符串中最长的回文字符串
   * 暴力枚举：找出所有的子字符串，判断是否是回文，并且记录长度
   * 动态规划：在上述的基础上，如果能知道某个子串是回文字符串，那么也可以推到出更长的回文字符串
   * dp[i][j] 表示i-j索引的子串，是不是回文字符串
   * ***/
  public int getLongestPalindrome(String A, int n) {
    boolean[][] dp = new boolean[n][n];
    dp[0][0] = true;
    for(int i = 1; i<n; i++){
      dp[i][i] = true;
      dp[i][i-1] = true;
    }
    int max = 0;
    for(int i = 1; i<n; i++){
      for(int j = 0; j<i; j++){
        if( dp [j+1][i-1] && A.charAt(i) == A.charAt(j)){
          dp[j][i] = true;
          max = Math.max(max, i-j+1);
        } else{
          dp[j][i] = false;
        }
      }
    }
    return  max;
  }


    /***1328. 破坏回文串
     * ***/
  public String breakPalindrome(String palindrome) {
    if (palindrome == null || palindrome.length() <= 1) {
      return "";
    }
    char[] chars = palindrome.toCharArray();
    for(int i = 0; i < chars.length / 2; i++){
        if(chars[i]  != 'a'){
          chars[i] = 'a';
          String str = new String(chars);
          return str;
      }
    }
    return palindrome.substring(0, palindrome.length() - 1) + 'b';

  }
  /****1249. 移除无效的括号
   * ****/
//  public String minRemoveToMakeValid(String s) {
//
//  }

  //字典序排序后取第小
  public int solution(int n, int m){
    //大根堆
    PriorityQueue<Integer> pq = new PriorityQueue<>(((o1, o2) ->
    { //按照字典序重排大根堆
      String str1 = String.valueOf(o1);
      String str2 = String.valueOf(o2);
      if(str1.length() == str2.length()){
        return (int)o2 - o1;
      } else {
        for(int i = 0; i<Math.min(str1.length(), str2.length()); i++){
          if(str1.charAt(i) != str2.charAt(i)){
            return str2.charAt(i) - str1.charAt(i);
          }
        }
        return str2.length() - str1.length();
      }

    }
    ));
    for(int i = 1; i<=n; i++){
      pq.add(i);
      if(pq.size() > m){
        pq.poll();
      }
    }
    return pq.peek();

  }





    public static void main(String[] args){
    System.out.println("hello string");
    StringTest stringTest = new StringTest();
    int r  = stringTest.calculate("3/2");
    int res = stringTest.getLongestPalindrome("abc1234321ab", 12);



  }


}
