package JavaTest.thread;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * User: gaohan
 * Date: 2021/3/2
 * Time: 22:50
 */
public class MyRunableTest implements Runnable {

  private String command;

  public MyRunableTest(String s){
    this.command = s;
  }
  @Override
  public void run() {
    System.out.println(Thread.currentThread().getName() + "start.time =" + new Date());
    try {
      sleepThread();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    System.out.println(Thread.currentThread().getName() + "end.time =" + new Date());
  }

  private void sleepThread() throws InterruptedException {
    try{
      Thread.sleep(5000);
    } catch (InterruptedException e){
      e.printStackTrace();
    }
  }
}
