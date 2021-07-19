package JavaTest.thread;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: gaohan
 * Date: 2021/3/2
 * Time: 22:58
 */
public class ThreadPoolExecutorDemo {
  public static final int corrPoolSize = 5;
  public static final int maxPoolSize = 10;
  public static final int maxQueue  = 100;
  public static final long keepAliveTime = 1l;

  public static void main(String[] args) {
    ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(corrPoolSize,
      maxPoolSize,
      keepAliveTime,
      TimeUnit.SECONDS,
      new ArrayBlockingQueue<>(maxQueue),
      new ThreadPoolExecutor.CallerRunsPolicy()) ;//command+p

    for(int i = 0; i<10; i++){
      Runnable worker = new MyRunableTest("" + i);
      threadPoolExecutor.execute(worker); //执行每个线程
    }

    threadPoolExecutor.shutdown();//终止线程池,
    while(!threadPoolExecutor.isTerminated()){

    }
    System.out.println("Finished all threads");


  }
}
