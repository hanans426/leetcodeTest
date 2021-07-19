package JavaTest.thread;

import java.util.Collection;
import java.util.concurrent.*;

/**
 * User: gaohan
 * Date: 2021/3/2
 * Time: 14:54
 */
//用接口实现线程更好，因为单继承和多实现，并且继承整个Thread 开销有点大
public class TestThread {

  //通过实现Runnable接口，在通过实例化一个Thread, 调用Thread 的start
  public static class MyRunnable implements Runnable{
    @Override
    public void run() {
      System.out.println("hello runnable");
    }
  }

  //Callable 可以有返回值，返回值通过FutureTask 来封装
  public static class MyCallable implements Callable<Integer>{
    @Override
    public Integer call() throws Exception {
      return 1;
    }
  }

  //Thread类也是实现类runable 接口，所以也要实现run方法
  public static class MyThread extends Thread{
    @Override
    public void run() {
      System.out.println("hello thread");
    }
  }

  public  static void main(String[] args) throws ExecutionException, InterruptedException {
    MyRunnable instance = new MyRunnable();
    Thread thread1 = new Thread(instance);
    thread1.start(); // 调用start方法并进入就绪状态，分配到时间片后就可以运行了，start会进行相关准备工作，并执行run方法中的内容，直接调用run方法会把run方法当作一个main线程下的普通函数

    MyCallable inst1 = new MyCallable();
    FutureTask<Integer> ft = new FutureTask<>(inst1); //Callable 的返回值通过FutureTask 来封装
    Thread thread2 = new Thread(ft);
    thread2.start();
    System.out.println(ft.get());

    MyThread myThread = new MyThread();
    myThread.start();

    Singleton obj = Singleton.getInstance();
    obj.run();


  }

}
