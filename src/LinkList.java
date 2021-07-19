import java.util.HashMap;

/**
 * User: gaohan
 * Date: 2020/12/21
 * Time: 11:15
 * 链表，链表是空节点，或者有一个值和一个指向下一个链表的指针。常用递归的方法处理
 */
public class LinkList {

  /******160. 相交链表
   * 双指针法：链表A的指针访问到链表A的尾部时，令该指针从B的头部开始，链表B的指针访问到尾部时，令该指针从A的头部开始
   * 这样就能控制访问A B 两个链表的指针同时访问到交点
   * ******/

  public class ListNode {
     int val;
      ListNode next;
      ListNode(int x) {
          val = x;
          next = null;
     }
  }

  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode l1 = headA;
    ListNode l2 = headB;
    while (l1 != l2) {
      l1 = (l1 == null) ? headB : l1.next;
      l2 = (l2 == null) ? headA : l2.next;
    }
    return l1;
  }

  /***206. 反转链表
   * 1. 迭代法 采用两个指针，cur在前，pre 在后，把cur.next 指向pre,然后cur 和pre 都向前移动一个位置，直到cur 为空后，pre指向的就是最后一个元素
   * ****/

  public ListNode reverseList(ListNode head) {
    if(head == null || head.next == null){
      return  head;
    }
    ListNode pre = null;
    ListNode cur = head;

    while(cur != null){
      ListNode temp = cur.next;
      cur.next = pre;
      pre = cur;
      cur = temp;

    }
    return pre;

  }
 //用递归的方法解决，较难
  public ListNode reverseList1(ListNode head) {
    if(head == null || head.next == null){
      return  head;
    }
    ListNode cur = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return cur;
  }

  /****21. 合并两个有序链表
   * 递归法：
   * 迭代法: 最后不忘记把剩余的元素添加进来
   * ---------------important
   * *****/
  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l1 == null) return l2;
    if(l2 == null) return l1;
    if(l1.val < l2.val){
      l1.next = mergeTwoLists(l1.next, l2);
      return l1;
    } else {
      l2.next = mergeTwoLists(l1, l2.next);
      return l2;
    }

  }

  public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
    ListNode preHead = new ListNode(-1);
    ListNode prev = preHead;
    while(l1 != null && l2 != null){ //注意边界条件
      if(l1.val < l2.val){
        prev.next = l1;
        l1 = l1.next;
      } else {
        prev.next = l2;
        l2 = l2.next;
      }
      prev = prev.next;
    }

    prev.next = l1 == null? l2:l1;
    return preHead.next;


  }

  /****83. 删除排序链表中的重复元素
   * ****/
  public ListNode deleteDuplicates(ListNode head) {
    ListNode cur = head;
    while(cur != null && cur.next != null){  //z注意边界条件
      if(cur.val == cur.next.val){
        cur.next = cur.next.next;
      } else {
        cur = cur.next;
      }

    }
    return head;

  }

  public ListNode deleteDuplicates1(ListNode head) {
    if(head == null || head.next == null) return  head;
    head.next = deleteDuplicates(head.next);
    return head.val == head.next.val? head.next: head;

  }

  /****24. 两两交换链表中的节点
   * *****/
  public ListNode swapPairs(ListNode head) {
    ListNode preHead = new ListNode(-1);
    preHead.next = head;
    ListNode pre = preHead;
    while(pre.next != null && pre.next.next != null){
      ListNode l1 = pre.next;
      ListNode l2 = pre.next.next;
      ListNode next = l2.next;
      l1.next = next;
      l2.next = l1;
      pre.next = l2;
      pre = l1;

    }
    return preHead.next;

  }

  /*****328. 奇偶链表
   * *****/
  public ListNode oddEvenList(ListNode head) {
    if(head == null) {
      return head;
    }
    ListNode odd = head;
    ListNode even = head.next;
    ListNode evenHead = even; // 保存偶数链的头部节点
    while(even != null && even.next != null){
      odd.next = odd.next.next;
      odd = odd.next;  // odd 推进到下一个节点
      even.next  = even.next.next;
      even = even.next;  // even 推进到下一节点
    }
    odd.next = evenHead;
    return head;

  }

  /***25. K 个一组翻转链表
   * ***/
  public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null || head.next == null){
      return head;
    }
    ListNode dummy = new ListNode(0);
    dummy.next = head;

    ListNode pre = dummy; //当前要翻转链表的头节点的上一个节点
    ListNode end = dummy; //当前要翻转链表的尾节点

    while(end.next != null){ // 说明还有要翻转的
      for(int i = 0; i<k&&end != null; i++){
        end = end.next;
      }
      if(end == null){ // 说明剩下的不够了， 要跳出循环
        break;
      }
      ListNode next = end.next; //为了方便链接
      end.next = null;// 断开链表
      ListNode start = pre.next;
      pre.next = myReverse(start); //翻转后，start就是最后的元素了
      start.next = next;
      pre = start;
      end = pre;

    }
    return dummy.next;

  }
  private ListNode myReverse(ListNode head){
    if (head == null || head.next == null){
      return head;
    }
    ListNode pre = null;
    ListNode cur = head;
    while (cur != null){
     ListNode next = cur.next;
     cur.next = pre;
     pre = cur;
     cur = next;

    }
    return pre;
  }

  /***链表按奇数位和偶数位进行拆分
   * ***/
  public ListNode[] getList(ListNode head){
    if(head == null){
      return null;
    }
    ListNode head1 = new ListNode(-1);
    ListNode head2 = new ListNode(-1);
    ListNode cur1 = head1;
    ListNode cur2 = head2;
    int cnt = 1;
    while(head != null){
      if(cnt%2 == 1){ //奇数
        cur1.next = head;
        cur1 = cur1.next;
      }else {
        cur2.next = head;
        cur2 = cur2.next;
      }
      head = head.next;
      cnt +=1;
    }
    cur1.next = null;
    cur2.next = null;

   ListNode[] res = new ListNode[]{head1.next, head2.next};
    return res;
  }



}
