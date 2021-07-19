//import java.util.*;
//
//
// class TreeNode {
//    int val = 0;
//    TreeNode left = null;
//    TreeNode right = null;
//  }
//
//
//public class solution {
//  /**
//   *
//   * @param root TreeNode类 the root of binary tree
//   * @return int整型二维数组
//   */
//  public int[][] threeOrders (TreeNode root) {
//    // write code here
//    List<Integer> pre = new ArrayList<>();
//    List<Integer> in = new ArrayList<>();
//    List<Integer> post = new ArrayList<>();
//    preOrder(root, pre);
//    inOrder(root, in);
//    postOrder(root, post);
//
//    Integer[] p1 = pre.toArray(new Integer[0]);
//    return result;
//
//
//  }
//
//  //先序遍历
//  private void preOrder(TreeNode root, List<Integer> list){
//    if(root == null){
//      return;
//    }
//    list.add(root.val);
//    preOrder(root.left, list);
//    preOrder(root.right, list);
//  }
//
//  //中序遍历
//  private void inOrder(TreeNode root, List<Integer> list){
//    if(list == null){
//      return;
//    }
//    inOrder(root.left, list);
//    list.add(root.val);
//    inOrder(root.right, list);
//  }
//
//  private void postOrder(TreeNode root, List<Integer> list){
//    if(list == null){
//      return;
//    }
//    postOrder(root.left, list);
//    postOrder(root.right, list);
//    list.add(root.val);
//  }
//
//
//
//
//
//
//
//
//
//
//
//
//}

