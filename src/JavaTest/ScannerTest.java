package JavaTest;

import java.util.*;

/**
 * User: gaohan
 * Date: 2021/3/18
 * Time: 10:08
 */
public class ScannerTest {

  //求两个字符串的最长子序列
  // 首先求最长子序列的长度
  public String LCS (String str1, String str2) {
    // write code here
    int m = str1.length();
    int n = str2.length();
    int maxlen = 0;
    int end = 0;
    int[][] dp = new int[m+1][n+1];
    for(int  i = 0; i<=m; i++){
      dp[i][0] = 0;
    }
    for(int j = 0; j<=n; j++){
      dp[0][j] = 0;
    }
    for(int i = 1; i<=m;i++){
      for(int j = 1; j<=n; j++){
        if(str1.charAt(i-1) == str2.charAt(j-1)){
          dp[i][j] = dp[i-1][j-1]+1;
        } else{
          dp[i][j] = 0;
        }
        if(dp[i][j] > maxlen){
          maxlen = dp[i][j];
          end = j-1;
        }

      }

    }
    if(maxlen == 0){
      return "-1";
    } else {
      return str2.substring(end - maxlen +1, end+1);
    }
  }

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

  //import java.util.*;
//public class Main{
//
//    public int solution(int[] p, int n){
//        int res = 0;
//        int[] cnt = new int[n];//记录一个房间内跳过的次数
//        int i = 0; // 表示目前房间的索引
//        while(i<n){
//            if( cnt[i]%2 == 0){ //为偶数时，说明这次是第奇数次跳进去
//                cnt[i] += 1;
//                i = p[i] - 1;
//            } else {
//                cnt[i] += 1;
//                i += 1;
//            }
//
//        }
//
//        //i==n 时说明已经到达了最后一个房间，求出cnt 中的总和
//        for(int j = 0; j<n; j++){
//            res += cnt[j];
//            if(res >1000000007 ){
//                res = res%1000000007;
//
//            }
//        }
//        return res;
//
//    }
//
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        int nums = sc.nextInt();
//        sc.nextLine();
//        String[] strs = sc.nextLine().split(" ");
//        int[] p = new int[nums];
//        for(int i = 0; i<nums; i++){
//            p[i] = Integer.parseInt(strs[i]);
//        }
//        Main main = new Main();
//        int res = main.solution(p, nums);
//        System.out.println(res);
//    }
//}

  public boolean solution(long n, long k, long d1, long d2){
    long[] x = new long[]{k-2*d1-d2, k+ 2* d1+d2, k-2*d1+d2, k+ 2*d1-d2};
    for(int i = 0; i<x.length; i++){
      if(x[i]%3 != 0 ){
        continue;
      }

      long a1 = x[i]/3;
      long a2 = 0;
      long a3 = 0;
      if( i == 0){
        a2 = a1 + d1;
        a3 = a2 + d2;
      } else if( i == 1){
        a2 = a1 - d1;
        a3 = a2 - d2;
      } else if( i == 2){
        a2 = a1 + d1;
        a3 = a2 - d2;
      } else {
        a2 = a1 - d1;
        a3 = a2 + d2;
      }
      if(a1 <= n/3 && a2 <= n/3 && a3 <= n/3 && a1>=0 && a2 >=0 && a3 >=0){
        return true;
      } else {
        continue;
      }

    }
    return false;
  }

  class Solution {
    List<List<Integer>> edges;
    int[] visited;
    boolean valid = true;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
      edges = new ArrayList<List<Integer>>();
      for (int i = 0; i < numCourses; ++i) {
        edges.add(new ArrayList<Integer>());
      }
      visited = new int[numCourses];
      for (int[] info : prerequisites) {
        edges.get(info[1]).add(info[0]);
      }
      for (int i = 0; i < numCourses && valid; ++i) {
        if (visited[i] == 0) {
          dfs(i);
        }
      }
      return valid;
    }

    public void dfs(int u) {
      visited[u] = 1;
      for (int v: edges.get(u)) {
        if (visited[v] == 0) {
          dfs(v);
          if (!valid) {
            return;
          }
        } else if (visited[v] == 1) {
          valid = false;
          return;
        }
      }
      visited[u] = 2;
    }
  }


  public static int[][] solution(int[][] arrs){
    //先按照x的坐标对arrs 进行排序
    Arrays.sort(arrs, ((o1, o2) -> o1[0] - o2[0]));
    List<List<Integer>> res = new ArrayList<>();

    int maxY = arrs[arrs.length -1][1];
    List<Integer> first = new ArrayList<>();
    first.add(arrs[arrs.length -1][0]);
    first.add(arrs[arrs.length -1][1]);
    res.add(first);
    for(int i = arrs.length - 2; i>=0; i--){
      List<Integer> l = new ArrayList<>();
      if(arrs[i][1] > maxY){
        l.add(arrs[i][0]);
        l.add(arrs[i][1]);
        res.add(l);
        maxY = arrs[i][1];
      }
    }
    int[][] r = new int[res.size()][2];
    for(int i = res.size() - 1; i>=0; i--){
      r[res.size() - i][0] = res.get(i).get(0);
      r[res.size() - i][1] = res.get(i).get(1);
    }
    return r;

  }


  public static void main(String[] args) {
    ScannerTest test = new ScannerTest();
    int res = test.solution(200, 25);
    System.out.println(res);
    Scanner sc = new Scanner(System.in);
    int nums = Integer.parseInt(sc.nextLine());
    String[] arrs = new String[nums];
    for (int i = 0; i < nums; i++) {
      arrs[i] = sc.nextLine();
    }
    sc.close();
    for (int i = 0; i < nums; i++) {
      String[] strs = arrs[i].split(" ");
      String s1 = strs[0];
      String s2 = strs[1];
      String lcs = test.LCS(s1, s2);
      System.out.println(lcs);
      //int s1 = Integer.parseInt(strs[0]);
    }


  }
}
