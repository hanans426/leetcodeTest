package JavaTest.thread;

/**
 * User: gaohan
 * Date: 2021/3/3
 * Time: 15:33
 */
public class SynchronizedDemo {
  public void method(){
    synchronized (this){
      System.out.println("helle sychronized ");
    }
  }
  public static void main(String[] args) {

  }
}
