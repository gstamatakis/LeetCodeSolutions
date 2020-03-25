import LeetCode._
import org.scalatest.FunSuite

class LeetTests extends FunSuite {
  //1
  test("TwoSum") {
    assert(twoSum(Array(), 999) === Array())
    assert(twoSum(Array(1, 2), 3) === Array(0, 1))
    assert(twoSum(Array(2, 7, 11, 15), 9) === Array(0, 1))
  }

  //No 5
  test("longestPalindrome") {
    assert(longestPalindrome("babad") === "bab")
    assert(longestPalindrome("cbbd") === "bb")
  }


  /**
   * 38. Count and Say
   *
   * Example 1:
   *
   * Input: 1
   * Output: "1"
   * Explanation: This is the base case.
   *
   *
   * Example 2:
   * Input: 4
   * Output: "1211"
   * Explanation: For n = 3 the term was "21" in which we have two groups "2" and "1", "2" can be
   * read as "12" which means frequency = 1 and value = 2, the same way "1" is read as "11", so the
   * answer is the concatenation of "12" and "11" which is "1211".
   */
  test("Count and say") {
    assert(countAndSay(1) === "1")
    assert(countAndSay(4) === "1211")
  }

  /**
   * 26. Remove Duplicates from Sorted Array
   *
   * Example 2:
   *
   * Given nums = [0,0,1,1,1,2,2,3,3,4],
   *
   * Your function should return length = 5, with the first five
   * elements of nums being modified to 0, 1, 2, 3, and 4 respectively.
   *
   * It doesn't matter what values are set beyond the returned length.
   */
  test("Remove duplicates") {
    assert(removeDuplicates(Array[Int](1, 1, 2)) === 2)
    assert(removeDuplicates(Array[Int](0, 0, 1, 1, 1, 2, 2, 3, 3, 4)) === 5)
  }

  /**
   * Example:
   * Input: nums = [1,2,3]
   * Output:
   * [
   * [3],
   * [1],
   * [2],
   * [1,2,3],
   * [1,3],
   * [2,3],
   * [1,2],
   * []
   */
  test("Subsets") {
    val target = List(List(), List(3), List(2), List(3, 2), List(1), List(3, 1), List(2, 1), List(3, 2, 1))
    assert(subsets(Array[Int](1, 2, 3)) == target)
  }


  //118
  test("Pascal's Triangle") {
    assert(pascalTriangle(5) ===
      List(List(1), List(1, 1), List(1, 2, 1), List(1, 3, 3, 1), List(1, 4, 6, 4, 1)))
    assert(pascalTriangle(1) === List(List(1)))
    assert(pascalTriangle(500).size == 500)
  }

}