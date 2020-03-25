
import scala.annotation.tailrec
import scala.collection.mutable

object LeetCode extends App {
  /**
   * No 1
   *
   * Given an array of integers, return indices of the two numbers such that they add up to a specific target.
   * You may assume that each input would have exactly one solution, and you may not use the same element twice.
   *
   * Runtime: 264 ms, faster than 99.88% of Scala online submissions for Two Sum.
   * Memory Usage: 48.5 MB, less than 100.00% of Scala online submissions for Two Sum.
   *
   * @param nums   Input array of nums.
   * @param target The target sum.
   * @return An array of INDICES of the selected array numbers.
   */
  def twoSum(nums: Array[Int], target: Int): Array[Int] = {
    if (nums.length == 0 || nums.length == 1) return Array()
    if (nums.length == 2) return Array(0, 1)
    val lookup = mutable.Map[Int, Int]()
    for (i <- nums.indices) {
      val el = nums(i)
      val res = target - el
      if (lookup.contains(el)) {
        return Array(lookup(el), i)
      } else {
        lookup.put(res, i)
      }
    }
    Array()
  }

  /**
   * No 5
   *
   * Given a string s, find the longest palindromic substring in s.
   * You may assume that the maximum length of s is 1000.
   *
   * @param s Input string.
   * @return A string with the longest palindrome.
   */
  def longestPalindrome(s: String): String = {
    @tailrec
    def maxAt(startIndex: Int, endIndex: Int): String = {
      if (startIndex < 0 || endIndex > s.length - 1 || s.charAt(startIndex) != s.charAt(endIndex))
        s.substring(startIndex + 1, endIndex)
      else
        maxAt(startIndex - 1, endIndex + 1)
    }

    @tailrec
    def _lp(st: Int, largestYet: String): String = {
      if (st == s.length) largestYet
      else {
        val largestOdd = maxAt(st - 1, st + 1)
        val largestEven = maxAt(st, st + 1)

        if (largestOdd.length > largestEven.length && largestOdd.length > largestYet.length)
          _lp(st + 1, largestOdd)
        else if (largestEven.length > largestOdd.length && largestEven.length > largestYet.length)
          _lp(st + 1, largestEven)
        else
          _lp(st + 1, largestYet)
      }
    }

    _lp(0, "")
  }


  /**
   * 26. Remove Duplicates from Sorted Array
   * Given a sorted array nums, remove the duplicates in-place such that each element appear only
   * once and return the new length.
   *
   * Do not allocate extra space for another array, you must do this by modifying the input array
   * in-place with O(1) extra memory.
   *
   * Example 1:
   * Given nums = [1,1,2],
   * Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.
   *
   * It doesn't matter what you leave beyond the returned length.
   */
  def removeDuplicates(nums: Array[Int]): Int = {
    @tailrec
    def check(as: Array[Int], i: Int, j: Int, c: Int): Int = {
      if (j >= as.length) c
      else if (as(i) == as(j)) check(as, i, j + 1, c)
      else {
        as(i + 1) = as(j)
        check(as, i + 1, j + 1, c + 1)
      }
    }

    if (nums.isEmpty) 0
    else check(nums, 0, 0, 1)
  }

  /**
   * 38. Count and Say
   *
   * The count-and-say sequence is the sequence of integers with the first five terms as following:
   *
   * 1.     1
   * 2.     11
   * 3.     21
   * 4.     1211
   * 5.     111221
   * 1 is read off as "one 1" or 11.
   * 11 is read off as "two 1s" or 21.
   * 21 is read off as "one 2, then one 1" or 1211.
   *
   * Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence. You can do so recursively, in other words from the previous member read off the digits, counting the number of digits in groups of the same digit.
   *
   * Note: Each term of the sequence of integers will be represented as a string.
   */
  def countAndSay(n: Int): String = {

    def toInt(c: Char): Int = c.toInt - '0'.toInt

    @tailrec
    def say(x: String, i: Int, ls: List[(Int, Int)], c: (Int, Int)): List[(Int, Int)] =
      if (i >= x.length) c :: ls
      else if (toInt(x(i)) == c._2) say(x, i + 1, ls, (c._1 + 1, c._2))
      else say(x, i + 1, c :: ls, (1, toInt(x(i))))


    def concat(ls: List[(Int, Int)]): String =
      ls.reverse.foldLeft("") {
        case (s, (a, b)) => s + a.toString + b.toString
      }

    @tailrec
    def cas(n: Int, x: Int, s: String): String =
      if (x == n) s
      else {
        val s0 = toInt(s(0))
        val t = concat(say(s, 0, Nil, (0, s0)))
        cas(n, x + 1, t)
      }

    cas(n, 1, "1")
  }

  //# 42. Trapping Rain Water
  def trap(height: Array[Int]): Int = {
    val stack = new mutable.Stack[Int]()
    var waterTrapped = 0

    for (i <- height.indices) {
      while (stack.nonEmpty && height(i) > height(stack.top)) { // With water trapped.
        val bottomHeight = height(stack.pop())
        if (stack.nonEmpty) {
          val leftHeight = height(stack.top)
          val width = i - stack.top - 1
          waterTrapped += width * (scala.math.min(leftHeight, height(i)) - bottomHeight)
        }
      }
      stack.push(i)
    }

    waterTrapped
  }

  /**
   * 78. Subsets
   * Given a set of distinct integers, nums, return all possible subsets (the power set).
   *
   * Note: The solution set must not contain duplicate subsets.
   * ]
   */
  def subsets(nums: Array[Int]): List[List[Int]] = {

    def iter(nums: List[Int], index: Int): List[List[Int]] = {
      if (nums.size == index) {
        List(Nil)
      } else {
        val allss: List[List[Int]] = iter(nums, index + 1)
        val newss: List[List[Int]] = for {
          ss <- allss
        } yield {
          ss :+ nums(index)
        }

        allss ++ newss
      }
    }

    iter(nums.toList, 0)
  }


  /**
   * No 118
   *
   * Creates a Pascal's Triangle up to the given row number.
   *
   * Runtime: 204 ms, faster than 96.59% of Scala online submissions for Pascal's Triangle.
   * Memory Usage: 42.6 MB, less than 100.00% of Scala online submissions for Pascal's Triangle.
   *
   * @param numRows The number of rows
   * @return A List of Lists with each row representing the Triangle's row at the given depth.
   **/
  def pascalTriangle(numRows: Int): List[List[Int]] = {
    def constructFrom(prevRow: List[Int]): List[Int] = prevRow match {
      case x1 :: x2 :: xs => (x1 + x2) :: constructFrom(x2 :: xs)
      case _ :: Nil => List(1)
      case Nil => List()
    }

    def genFromPrev(n: Int): List[List[Int]] =
      if (n == 1) {
        List(List(1))
      } else {
        val prev = genFromPrev(n - 1)
        (1 :: constructFrom(prev.head)) :: prev
      }

    if (numRows == 0) Nil else genFromPrev(numRows).reverse
  }


}