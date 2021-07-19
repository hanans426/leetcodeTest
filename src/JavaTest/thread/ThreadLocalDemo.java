package JavaTest.thread;

import java.text.SimpleDateFormat;
import java.util.Random;

/**
 * User: gaohan
 * Date: 2021/3/2
 * Time: 23:22
 */
//ThreadLocal 是一个线程内存放线程的私有遍历的地方，每一个线程的ThreadLocal与其他线程的相互隔离
public class ThreadLocalDemo implements Runnable{
  public static final ThreadLocal<SimpleDateFormat> formatter = ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyyMMdd HHmm"));

  @Override
  public void run() {
    System.out.println(Thread.currentThread().getName() + "default Formatter = " + formatter.get().toPattern());
    try {
      Thread.sleep(500);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    formatter.set(new SimpleDateFormat());
    System.out.println(Thread.currentThread().getName() + "formatter = " + formatter.get().toPattern());

  }

  public static void main(String[] args) throws InterruptedException {
    ThreadLocalDemo threadLocalDemo = new ThreadLocalDemo();
    for(int i = 0; i<10; i++){
      Thread t = new Thread(threadLocalDemo,""+i);
      Thread.sleep(100);
      t.start();
    }
  }
}
