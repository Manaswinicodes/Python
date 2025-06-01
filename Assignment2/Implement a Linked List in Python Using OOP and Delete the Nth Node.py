class Tony:
    def __init__(self, data):
        self.data = data
        self.next = None

# LinkedList class Avengers
class Avengers:
    def __init__(self):
        self.head = None

    # Add node to end: add_hero
    def add_hero(self, hero_name):
        new_hero = Tony(hero_name)
        if not self.head:
            self.head = new_hero
            return
        last_hero = self.head
        while last_hero.next:
            last_hero = last_hero.next
        last_hero.next = new_hero

    # Print list method: show_avengers
    def show_avengers(self):
        if not self.head:
            print("Avengers list is empty!")
            return
        current = self.head
        while current:
            print(current.data, end=" -> " if current.next else "\n")
            current = current.next

    # Delete nth node (1-based): delete_hero
    def delete_hero(self, n):
        if not self.head:
            raise Exception("Can't delete from empty Avengers list!")

        if n <= 0:
            raise Exception("Index should be 1 or greater!")

        current = self.head

        # If head to be deleted
        if n == 1:
            self.head = current.next
            return

        # Find previous node of the node to be deleted
        prev = None
        count = 1
        while current and count < n:
            prev = current
            current = current.next
            count += 1

        if not current:
            raise Exception("Index out of range!")

        # Delete current node
        prev.next = current.next

# Test the Avengers linked list
if __name__ == "__main__":
    avengers = Avengers()

    # Add some heroes
    avengers.add_hero("Iron Man")
    avengers.add_hero("Captain America")
    avengers.add_hero("Thor")
    avengers.add_hero("Black Widow")

    print("Initial Avengers List:")
    avengers.show_avengers()

    # Delete 3rd hero (Thor)
    avengers.delete_hero(3)
    print("After deleting 3rd hero (Thor):")
    avengers.show_avengers()

    # Delete 1st hero (Iron Man)
    avengers.delete_hero(1)
    print("After deleting 1st hero (Iron Man):")
    avengers.show_avengers()

    # Try deleting out of range index
    try:
        avengers.delete_hero(10)
    except Exception as e:
        print("Error:", e)

    # Try deleting from empty list
    empty_team = Avengers()
    try:
        empty_team.delete_hero(1)
    except Exception as e:
        print("Error:", e)


##Output:- 
Initial Avengers List:
Iron Man -> Captain America -> Thor -> Black Widow
After deleting 3rd hero (Thor):
Iron Man -> Captain America -> Black Widow
After deleting 1st hero (Iron Man):
Captain America -> Black Widow
Error: Index out of range!##
Error: Can't delete from empty Avengers list!
