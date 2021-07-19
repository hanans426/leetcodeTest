package JavaTest;

import java.util.Objects;

/**
 * User: gaohan
 * Date: 2021/3/1
 * Time: 23:02
 */
public class TestHash {

  static class TestNode{
    int v = 0;

    public TestNode(int v) {
      this.v = v;
    }

    public void setV(int v) {
      this.v = v;
    }

    public int getV() {
      return v;
    }

    //coommand +n

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof TestNode)) return false;
      TestNode testNode = (TestNode) o;
      return v == testNode.v;
    }

    @Override
    public int hashCode() {
      return Objects.hash(v);
    }

    @Override
    public String toString() {
      return "TestNode{" +
        "v=" + v +
        '}';
    }

    @Override
    protected void finalize() throws Throwable {
      super.finalize();
    }
  }

  public static void main(String[] args) {

    print(); //command+control +m 提取代码为方法
    
  }

  private static void print() {
    System.out.println("s");
  }
}
