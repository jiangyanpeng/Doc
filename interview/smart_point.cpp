#include <functional>
#include <iostream>
#include <memory>

// 如何实现一个类，在父类没有虚析构函数的情况下，通过父类指针析构子类对象？
class A {
public:
  A() {}

  //   virtual这是正确的写法，但是可以通过其他方式实现此功能
  //   virtual ~A() { std::cout << "A的析构" << std::endl; }

  ~A() { std::cout << "A的析构" << std::endl; }
};

class B : public A {
public:
  B() {}
  ~B() { std::cout << "B的析构" << std::endl; }
};

/*
template <class T>
class manager_ptr : public manager
{
public:
    manager_ptr(T p): ptr(p) {}
    ~manager_ptr() { delete ptr; }
private:
    T ptr;
};

template <class T>
class _shared_ptr
{
public:
    template<class Y>
    _shared_ptr(Y* p) { ptr_manager = new manager_ptr<Y*>(p); }
    ~_shared_ptr() { delete ptr_manager; }
private:
    manager* ptr_manager;
};
*/
// 分析：
// 为什么智能指针对象，销毁的时候，在父类析构函数非虚的情况下，能够先执行子类的析构函数，再执行父类的析构函数。
// 1.类A没有虚函数，对象在内存中，根本就没有虚指针这个信息，更没有虚函数表。所以，通过对象的底层信息找到虚函数的地址无法实现。
// 2.manager_ptr这个类的对象，持有的指针类型是B*，在_shared_ptr销毁的时候，执行delete
// ptr_manager，将访问manager的析构函数，由于这个析构函数是虚函数，执行manager_ptr的析构函数。由于manager_ptr持有的指针类型是B*，
// 因此，执行的是类B的析构函数。执行子类的析构函数会自动调用父类的析构函数,其中manager_ptr相当于类型擦除
// 3.考虑采用利用模板类型参数匹配,lambda加function来做类型擦除

template <typename T> class MySharedV2 {
public:
  template <typename T2> MySharedV2(T2 *p) {
    data_ = p;
    deleter_ = [p]() { delete p; };
  }
  ~MySharedV2() { deleter_(); }
  T *operator->() { return data_; }

private:
  std::function<void()> deleter_;
  T *data_ = nullptr;
};

int main() {
  // 1. 这是方式调用是无法调用子类的析构函数
  //   A *ptr = new B;
  //   delete ptr;

  // 2. 采用智能指针的方式可以正确调用
  //   std::shared_ptr<A> ptr(new B);

  // 3.采用function实现的方案
  MySharedV2<A> ptr(new B);
  return 0;
}