import java.util.*;

/**
 * User: gaohan
 * Date: 2021/2/21
 * Time: 20:53
 */
public class Interview {
  public int collatz(){
    int count = 0;
    long n = 0l;
    int max = 0;
    int num  =0;
    for(int i = 2; i < 1000000; i++){
      n = i;
      while(n != 1){
        if(n%2 == 0){
          n = n/2;
          count++;
        } else {
          n = 3*n +1;
          count++;
        }

      }
      if(count + 1 > max){
        max = count +1;
        num = i;
      }
      count= 0;

    }
    return max;

  }
  // 用LinkedList 实现Set的数据结构
  class SetByLinkedList{
    private LinkedList<Integer> list;

    public SetByLinkedList(){
      list = new LinkedList<>();
    }

    //O(n)的时间复杂度，因为其中包含检验元素是否已经存在
    public void add(int val){
      if(!list.contains(val)){ //以此实现set 内的元素不重复
        list.add(val);
      }
    }
    //O(n)的时间复杂度
    public void remove(int val){
      list.remove(val);
    }

    public int getSize(){
      return list.size();
    }
    //O(n)的时间复杂度
    public boolean contains(int val){
      return list.contains(val);
    }
  }
  //用二叉搜索树实现Set, 在时间上进行优化的同时，也能使Set内部保持有序
  class SetByBST{
    BST bst;
    public SetByBST(){
      bst = new BST();
    }
    //O（logn）
    public void add(int val){
      bst.add(val);
    }
    //O（logn）
    public boolean contain(int val){
      return bst.contains(val);
    }
    //O（logn）
    public void remove(int val){
      bst.remove(val);
    }
    public int getSize(){
      return bst.size;
    }
    public boolean isEmpty(){
      return bst.isEmpty();
    }

  }


  class BST{
   private class TreeNode{
      int val;
      TreeNode left;
      TreeNode right;
      public TreeNode(int v){
        val = v;
      }
    }

    TreeNode root;
    int size;
    public BST(){
      root = null;
      size = 0;
    }

    private TreeNode add(TreeNode root, int val){
      if(root == null){
        size++;
        return new TreeNode(val);
      }
      if(val < root.val){
        root.left = add(root.left, val);
      } else if(val > root.val){
        root.right = add(root.right, val);
      }
      return root;
    }

    public void add(int val){
      root = add(root, val);
    }


    public boolean contains(int val){
      return contains(root, val);

    }
    public boolean contains(TreeNode root, int val){
      if(root == null){
        return false;
      }
      if(val==root.val){
        return true;
      } else if(val > root.val){
        return contains(root.right, val);
      } else {
        return contains(root.left, val);
      }

    }

    public void remove(int v){
      root = remove(root, v);
    }

    public TreeNode remove(TreeNode root,int val){
      if(root == null){
        return null;
      }
      if(val > root.val){
        root.right = remove(root.right, val);
      } else if(val < root.val){
        root.left = remove(root.left, val);
      } else { // 要删除的是根节点时
        if(root.left == null){
          size--;
          TreeNode right = root.right;
          root.right = null;
          return right;
        }
        if(root.right == null){
          size--;
          TreeNode left = root.left;
          root.left = null;
          return left;
        }
        TreeNode node = getMin(root.right); //找到右子树的最小节点，就是删掉节点的下一节点
        node.right = removeMin(root.right);
        node.left = root.left;
        root.left = root.right = null;
        return node;

      }
      return null;
    }
    public TreeNode getMin( TreeNode root){
      if(size == 0){
        return null;
      }
      if(root.left == null){
        return root;
      } else {
        return getMin(root.left);
      }

    }
    public TreeNode removeMin(TreeNode root){
      if(root.left == null){
        size--;
        TreeNode right = root.right;
        root.right = null;
        return right;
      }
      root.left = removeMin(root.left);
      return root;

    }
    public int getSize(){
      return size;
    }
    public  boolean isEmpty(){
      return size == 0;
    }
  }

  /***
   * <html>
   *   <div>
   *     <div>
   *
   *     </div>
   *
   *     <button>1
   *     </button>
   *     <button>2
   *     </button>
   *
   *   </div>
   *   <div>
   *     <button>3
   *     </button>
   *   </div>
   * <html/>
   *
   * selector("div" "button")
   * ***/
  class Node{
    String tag;
    List<Node> children;
  }
//  public List<Node> findNodes(Node root, List<String> selectors){
//
//    List<Node> res = new ArrayList<>();
//
//    List<Node> nodeList = root.children; //option + command + vv
//    //shift + F6
//    if(nodeList.isEmpty() || selectors.isEmpty()){
//     // return res;
//    }
//
//    for(int i = 0; i<selectors.size(); i++){
//      if(root.tag.equals(selectors.get(i))){
//        List<Node> ch = nodeList;
//        selectors.remove(i);
//        for(Node c: ch){
//          findNodes(c, selectors);
//        }
//      } else {//没有匹配上的话，则在孩子节点中寻找
//        List<Node> ch = nodeList;
//        for(Node c: ch){
//          findNodes(c, selectors);
//        }
//
//      }
//    }
//
//  }
  //command +x 删除

  private  List<Node> findNodes(Node root, ArrayList<String> selectors) {
    Node resultNode = null;
    for (String selector : selectors) {
      String tag = root.tag;
      if (tag.equals(selector)) {
        resultNode = new Node();
        resultNode.tag = tag;
      }
    }

    List<Node> children = root.children;
    if (children != null && children.size() > 0) {
      List<Node> resultChildren = new LinkedList<Node>();
      for (Node child : children) {
        List<Node> nodes = findNodes(child, selectors);
        if (nodes != null) {
          resultChildren.addAll(nodes);
        }
      }
      if (resultChildren.size() > 0 && resultNode != null) {
        resultNode.children = resultChildren;
      }
    }

    if (resultNode == null) {
      return null;
    } else {
      ArrayList<Node> result = new ArrayList<Node>();
      result.add(resultNode);
      return result;
    }
  }

  public static void main(String[] args){
    System.out.println("hello interview");
    Interview interview = new Interview();
    int res = interview.collatz();
    System.out.println(res);
  }
}
