////import java.util.*;
////public class Main{
////    public static int solution(String str, String tar){
//////        int cnt = 1;
//////        int tail = 0;
//////        int len = tar.length();
//////        int dup = 0;
//////        HashMap<Character, Integer> hashMap = new HashMap<>();
//////        for(int i = 0; i<str.length(); i++){
//////            hashMap.put(str.charAt(i), i+1);
//////        }
//////        int pre = -1;
//////        for(int j = 0; j<tar.length(); j++){
//////            int cur = hashMap.get(tar.charAt(j));
//////            if(cur > pre){
//////                pre = cur;
//////            } else if(cur< pre) {
//////                cnt += 1;
//////              //  pre = -1;
//////            } else {
//////                dup++;
//////            }
//////            if(j == tar.length() - 1){
//////                tail = hashMap.get(tar.charAt(j));
//////            }
//////        }
//////        int res =  26 * cnt - (len - dup) - (26 - tail );
//////        return res;
//////    }
//////
//////    public static void main(String[] args){
//////        Scanner sc = new Scanner(System.in);
//////        String str = sc.nextLine();
//////        String tar = sc.nextLine();
//////        int res = solution(str, tar);
//////        System.out.println(res);
//////
//////    }
//////}
////
////
////import java.util.*;
////
////public class Main{
////    public static int solution(int[] a, int[] b, int n, int m){
////        int res = Integer.MAX_VALUE;
////        HashMap<Integer, Integer> mapA = new HashMap<>();
////        HashMap<Integer, Integer> mapB = new HashMap<>();
////        Set<Integer> setA= new HashSet<>();
////        Set<Integer> setB= new HashSet<>();
////
////        for(int i = 0; i<n; i++){
////            int cntA = mapA.getOrDefault(a[i], 0);
////            mapA.put(a[i], cntA+1);
////
////            int cntB = mapB.getOrDefault(b[i], 0);
////            mapB.put(b[i], cntB+1);
////
////            setA.add(a[i]);
////            setB.add(b[i]);
////        }
////        int cur = a[0];
////        for(int i: setB){
////            int tmp = 0;
////            boolean f = true;
////            if( i >= cur ){
////                tmp = i - cur;
////            } else {
////                tmp = i + m-cur;
////            }
////            for(int j: setA){
////                int tmpA = (j + tmp) % m;
////                if(!setB.contains(tmpA) || mapA.get(tmpA) != mapB.get(j)){
////                    f = false;
////                    break;
////                }
////
////            }
////            if(f) {
////                res = Math.min(res, tmp);
////            }
////
////        }
////        return res;
////
////    }
////
////    public static void main(String[] args){
////        Scanner sc = new Scanner(System.in);
////        int n = sc.nextInt();
////        int m = sc.nextInt();
////        sc.nextLine();
////        String[] strA = sc.nextLine().split(" ");
////        String[] strB = sc.nextLine().split(" ");
////        int[] a = new int[n];
////        int[] b = new int[n];
////        for(int i = 0; i<n; i++){
////            a[i] = Integer.parseInt(strA[i]);
////            b[i] = Integer.parseInt(strB[i]);
////        }
////        int res = solution(a, b, n, m);
////        System.out.println(res);
////
////
////    }
////}
////
////
////
////import java.util.*;
////
////public class Main{
////    public static int solution(int[] a, int[] b, int n, int m){
////        int res = Integer.MAX_VALUE;
////        HashMap<Integer, Integer> mapA = new HashMap<>();
////        HashMap<Integer, Integer> mapB = new HashMap<>();
////        Set<Integer> setA= new HashSet<>();
////        Set<Integer> setB= new HashSet<>();
////
////        for(int i = 0; i<n; i++){
////            int cntA = mapA.getOrDefault(a[i], 0);
////            mapA.put(a[i], cntA+1);
////
////            int cntB = mapB.getOrDefault(b[i], 0);
////            mapB.put(b[i], cntB+1);
////
////            setA.add(a[i]);
////            setB.add(b[i]);
////        }
////        int cur = a[0];
////        for(int i: setB){
////            int tmp = 0;
////            boolean f = true;
////            if( i >= cur ){
////                tmp = i - cur;
////            } else {
////                tmp = i + m-cur;
////            }
////            for(int j: setA){
////                int tmpA = (j + tmp) % m;
////                if(!setB.contains(tmpA)){
////                    f = false;
////                    break;
////                }
////            }
////            if(f){
////                res = Math.min(res, tmp);
////            }
////
////        }
////        return res;
////
////    }
////
////    public static void main(String[] args){
////        Scanner sc = new Scanner(System.in);
////        int n = sc.nextInt();
////        int m = sc.nextInt();
////        sc.nextLine();
////        String[] strA = sc.nextLine().split(" ");
////        String[] strB = sc.nextLine().split(" ");
////        int[] a = new int[n];
////        int[] b = new int[n];
////        for(int i = 0; i<n; i++){
////            a[i] = Integer.parseInt(strA[i]);
////            b[i] = Integer.parseInt(strB[i]);
////        }
////        int res = solution(a, b, n, m);
////        System.out.println(res);
////
////
////    }
////}
//
//import java.util.*;
//public class Main{
//    static  int res1 = 0;
//    static  int res2 = 0;
//    public static int solution2(int[] n, int flag){
//        Deque<Integer> path = new ArrayDeque<>();
//        dfs(0,path, n,1, flag );
//        return flag==1?res1:res2;
//    }
//    private static void dfs(int start, Deque<Integer> path, int[] arr, int total, int flag){
////        if(path.size()==arr.length){
////            return;
////        }
//        if(path.size() >= 1){
//            if(total % 2 == 1 && flag == 1){
//                res1++;
//            } else if(total %2 == 0 && flag == 2){
//                res2++;
//            }
//        }
//        for(int i = start; i<arr.length; i++){
//            path.addLast(arr[i]);
//            dfs(i+1, path, arr, total* arr[i], flag);
//            path.removeLast();
//        }
//    }
//
//    public int solution(int[] arr, int n){
//        if(n == 1){
//            return arr[0];
//        }
//        int[] dp = new int[n+1];
//        dp[0] = 0;
//        dp[1] = arr[0];
//        for (int i = 2; i <= n; i++) {
//            int s1 = dp[i-2] + arr[i];
//            int s2 = dp[i-1] + arr[i] -  (arr[i]+1)/2;
//            dp[i] = Math.max(s1,s2);
//        }
//        return dp[n];
//    }
//
//
//
//    public String solve (int[] nums) {
//        ArrayList<String> list = new ArrayList<>();
//        for (int i = 0; i < nums.length; i++) {
//            list.add(String.valueOf(nums[i]));
//        }
//        Collections.sort(list, new Comparator<String>() {
//            @Override
//            public int compare(String o1, String o2) {
//                return (o2 + o1).compareTo(o1 + o2);
//            }
//        });
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < list.size(); i++) {
//            sb.append(list.get(i));
//        }
//        return  sb.toString();
//
//    }
//
//    public static long solve1 (int[] A) {
//        // write code here
//        int max1 = Integer.MIN_VALUE; //最大
//        int max2 = Integer.MIN_VALUE; //第二大
//        int max3 = Integer.MIN_VALUE; //第三大
//        int min1 = Integer.MAX_VALUE; //最小
//        int min2 = Integer.MAX_VALUE; //第二小
//        if(A.length == 3){
//            return (long)A[0]*A[1]*A[2];
//        }
//
//        for(int num : A){
//            if(num > max1 ){
//                max3 = max2;
//                max2 = max1;
//                max1 = num;
//            } else if(num > max2){
//                max3 = max2;
//                max2 = num;
//            } else if(num > max3){
//                max3 = num;
//            }
//            if(num < min1){
//                min2 = min1;
//                min1 = num;
//            } else if(num < min2){
//                min2 = num;
//            }
//        }
//        return Math.max((long)max1*max2*max3, (long)min1*min2*max1);
//    }
//
//
//
//    public static void main(String[] args){
//        int[] A = {1,-5,-2,3};
//        long l = solve1(A);
////        Scanner sc = new Scanner(System.in);
////        int T = sc.nextInt();
////        sc.nextLine();
////        int n = sc.nextInt();
////        sc.nextLine();
////        int[] arr = new int[n];
////        String[] str = sc.nextLine().split(" ");
////        for(int i = 0; i<n; i++){
////            arr[i] = Integer.parseInt(str[i]);
////        }
//
//
////        Scanner sc = new Scanner(System.in);
////        int n = sc.nextInt();
////        int m = sc.nextInt();
////        sc.nextLine();
////        int[] arr = new int[n];
////        for(int i = 0; i<n; i++){
////            arr[i] = sc.nextInt();
////        }
////
////        for(int i = 0; i<m; i++){
////            sc.nextLine();
////            int t = sc.nextInt();
////            int l = sc.nextInt();
////            int r = sc.nextInt();
////            int[] nums = new int[r - l + 1];
////            for(int j = 0; j < nums.length; j++){
////                nums[j] = arr[ j + l];
////            }
////            int res = solution2(nums, t);
////            System.out.println(res);
////
////
////        }
//
//    }
//}
//
//
//
//
//
import java.util.*;
public class Main{
    public static int solution(int N, String numStr){
        int[] dp = new int[N];
        char[] arr = numStr.toCharArray();
        HashMap<Integer, Integer> map = new HashMap<>();
        dp[0] = 0;
        map.put(arr[0] - '0', 0);
        for (int i = 1; i < arr.length; i++) {
            if(map.containsKey(arr[i] - '0')){
                dp[i] = Math.min(dp[i-1] +1, map.get(arr[i]-'0')+1);
            } else {
                dp[i] = dp[i-1]+1;
                map.put(arr[i] - '0',dp[i]);
            }
        }
        return dp[N-1];
    }

    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        sc.nextLine();
        String str = sc.nextLine();
        int res = solution(n, str);
        System.out.println(res);

    }
}


