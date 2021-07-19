package exam;

import java.util.*;

/**
 * User: gaohan
 * Date: 2021/3/21
 * Time: 15:51
 */
public class Meituan {

  public static int solution1(String str, String tar){
    int cnt = 1;
    int tail = 0;
    int m = str.length();
    HashMap<Character, Integer> hashMap = new HashMap<>();
    for(int i = 0; i<str.length(); i++){
      hashMap.put(str.charAt(i), i+1);
    }
    int pre = hashMap.get(tar.charAt(0));
    int res = hashMap.get(tar.charAt(0)) - 1;
    for(int j = 1; j<tar.length() ; j++){
      int cur = hashMap.get(tar.charAt(j));
      if(cur > pre){
        res += cur - pre - 1;
        pre = cur;
      } else  {
        res += m - pre + cur - 1;
        pre = cur;
      }
    }

    return res;
  }

//  public static void main(String[] args){
//    Scanner sc = new Scanner(System.in);
//    String str = sc.nextLine();
//    String tar = sc.nextLine();
//    int res = solution1(str, tar);
//    System.out.println(res);
//  }



  public static int solution2(int[] a, int[] b, int n, int m){
    int res = Integer.MAX_VALUE;
    HashMap<Integer, Integer> mapA = new HashMap<>();
    HashMap<Integer, Integer> mapB = new HashMap<>();
    Set<Integer> setA= new HashSet<>();
    Set<Integer> setB= new HashSet<>();

    for(int i = 0; i<n; i++){
      int cntA = mapA.getOrDefault(a[i], 0);
      mapA.put(a[i], cntA+1);

      int cntB = mapB.getOrDefault(b[i], 0);
      mapB.put(b[i], cntB+1);

      setA.add(a[i]);
      setB.add(b[i]);
    }
    int cur = a[0];
    for(int i: setB){
      int tmp = 0;
      boolean f = true;
      if( i >= cur ){ // 此处只考虑了一次取模的情况
        tmp = i - cur;
      } else {
        tmp = i + m-cur;
      }
      for(int j: setA){
        int tmpA = (j + tmp) % m;
        if(!setB.contains(tmpA) || mapA.get(j) != mapB.get(tmpA)){
          f = false;
          break;
        }
      }
      if(f){
        res = Math.min(res, tmp);
      }

    }
    return res;

  }

  public static int solution3(int[] a, int[] b, int n, int m){
    int res = Integer.MAX_VALUE;
    HashMap<Integer, Integer> mapA = new HashMap<>();
    HashMap<Integer, Integer> mapB = new HashMap<>();

    for(int i = 0; i<n; i++){
      int cntA = mapA.getOrDefault(a[i], 0);
      mapA.put(a[i], cntA+1);

      int cntB = mapB.getOrDefault(b[i], 0);
      mapB.put(b[i], cntB+1);
    }
    int cur = a[0];

    for(Map.Entry<Integer, Integer> A : mapA.entrySet()){
      for(Map.Entry<Integer, Integer> B : mapB.entrySet()){
        int tmp = 0;
        if(A.getValue() == B.getValue()){
          tmp = B.getKey() >= A.getKey()? B.getKey() - A.getKey():B.getKey()+ m - A.getKey();
          res = Math.min(res, tmp);
        }
      }
    }
    return res;

  }

  public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    int m = sc.nextInt();
    sc.nextLine();
    String[] strA = sc.nextLine().split(" ");

    String[] strB = sc.nextLine().split(" ");
    int[] a = new int[n];
    int[] b = new int[n];
    for(int i = 0; i<n; i++){
      a[i] = Integer.parseInt(strA[i]);
      b[i] = Integer.parseInt(strB[i]);
    }
    int res = solution2(a, b, n, m);
    System.out.println(res);
  }
}
