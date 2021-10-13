#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>

template<typename T>
class thread_safe_queue {
  std::queue<T> queue_;
  mutable std::mutex mutex_;
 
  // Moved out of public interface to prevent races between this
  // and pop().
  bool empty() const {
    return queue_.empty();
  }
 
 public:
  thread_safe_queue() = default;
  thread_safe_queue(const thread_safe_queue<T> &) = delete ;
  thread_safe_queue& operator=(const thread_safe_queue<T> &) = delete ;
 
  thread_safe_queue(thread_safe_queue<T>&& other) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_ = std::move(other.queue_);
  }
 
  virtual ~thread_safe_queue() { }
 
  unsigned long size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }
 
  T pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }
    T tmp = queue_.front();
    queue_.pop();
    return tmp;
  }
 
  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
  }
};

#endif
