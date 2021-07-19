package JavaTest.thread;

/**
 * User: gaohan
 * Date: 2021/3/3
 * Time: 14:32
 */
public class Singleton implements Runnable {

  //饿汉类，迫不及待的实例化，开始的时候就初始化，可能会浪费内存，线程安全
//  private static Singleton instance = new Singleton(); //类加载的时候就实例化
//  private Singleton(){} //构造函数必须是私有的，才能保证不能被实例化
//  public static Singleton getInstance(){ //获取唯一的实例
//    return instance;
//  }
//  public void run(){
//    System.out.println("饿汉类");
//  }

  //懒汉式，在需要的时候再去实例化，是线程不安全的，如果多个线程同时去创建实例的话，会创造多个实例
//  private static Singleton instance;
//  private Singleton(){}
//  public static Singleton getInstance(){
//    if(instance == null){
//      instance = new Singleton();
//    }
//    return instance;
//  }
//  public void run(){
//    System.out.println("懒汉式不安全");
//  }

  //懒汉 线程安全
//  private static Singleton instance;
//  private Singleton(){}
//  public static synchronized Singleton getInstance(){ //同时只有一个线程可以访问
//    if(instance == null){
//      instance = new Singleton();
//    }
//    return instance;
//  }
//  public void run(){
//    System.out.println("懒汉式不安全");
//  }

  //双重校验锁实现对象单例（线程安全的),双锁机制，在多线程的情况下保持高性能
  private  volatile static Singleton instance; //用volatile 修饰
  private Singleton(){}
  public static synchronized Singleton getInstance(){ //同时只有一个线程可以访问
    if(instance == null){
      synchronized (Singleton.class){ //类对象加锁
        if(instance == null){
          instance = new Singleton();
          //此步骤分为了三步，
          //1. 为instance 分配内存空间
          //2。 初始化instance;
          //3. 将instance指向分配的内存空间
          //由于jvm的指令重排，可能顺序是1。3，2，可能在多线程的情况下出现返回还没初始化的实例
        }
      }
    }
    return instance;
  }
  public void run(){
    System.out.println("懒汉式不安全");
  }









  public static void main(String[] args) {
    Singleton obj = Singleton.getInstance();

    for(int i = 0; i<10; i++){
      Thread thread = new Thread(obj);
      thread.start();
    }


  }
}
