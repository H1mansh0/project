                                      Звіт з комп’ютерного проєкту на тему: ‘’Аналіз рекурентних рівнянь’’

 У даному проєкті ми навчилися знаходити розв’язки однорідних рекурентних рівнянь двома шляхами – аналітичним і logn-им.   Для виконання всіх завдань ми розподілиn-ли роботу наступним:

•	Атаманюк Владислав – 3-5 завдання, звіт

•	Віталій Палійчук – 3-5 завдання, презентація

•	Захар Когут – 1-2 завдання

•	Назар Тхір – 1-2 завдання

•	Ігор Іванишин – 1-2 завдання, презентація

 Перша частина проекту полягала в тому, щоб знайти загальний розв’язок рекурентного рівняння, а потім розв’язати систему лінійних рівнянь, щоб знайти коефіцієнти і отримати готову послідовність.

 Рекурентне рівняння функції отримують у вигляді списку коефіцієнтів полінома, у який перетворюється рекурентне рівняння після заміни аₙ = r^n і перенесення всіх членів в одну сторону. Коефіцієнти розміщені в порядку зростання степенів членів, біля яких вони стоять.
Для ефективного виконання всіх завдання проєкту ми розробили наступні функції:

- функції linear_equation(), quadratic_equation() і roots_of_polynomial()) - для лінійних і квадратних рівнянь пошук реалізований аналітично (через дискримінант). Для поліномів, степінь яких вищий за 2-ий, корені знаходяться підбором. Коли рівняння розв’язані, ми отримуємо загальний розв’язок рекурентного рівняння. Наступною задачею було складання та розв’язок системи лінійних рівнянь.

- find_c - аргументи це список коренів (coefs), отриманий на попередньому кроці, та список значень перших n членів шуканої послідовності (answers), де n - кількість коренів. Спочатку функція створює список списків - equations. Це і є наша система. Випадків є два: коли корені повторюються і коли всі корені різні. І той і інший покриваються функцією. В equations список містить коефіцієнти біля невідомих Cₙ, а останній член списку - це вільний член, отриманий з answers. Розв’язується система методом Крамера. Спочатку створюється матриця усіх коефіцієнтів рівняння - general_matrix - і шукається її визначник - det_gm. Визначник обчислюється за допомогою функції np.linalg.det() з бібліотеки NumPy. Потім через цикл кожен стовпець матриці замінюється на стовпець вільних членів (answers) і шукаються визначники кожної з отриманих матриць. Отримані визначники записуються в список і з них, діленням на визначник det_gm, ми отримуємо значення всіх невідомих коефіцієнтів С. Коли усі корені різні, коефіцієнти С повертаються списком, у якому кожен коефіцієнт поелементно відповідає кореню зі списку coefs переданого функції. Коли ж є повторювані корені, список coefs потрібно спочатку посортувати за зростанням, причому спочатку ставити невід’ємні значення. Тоді повернутий список коефіцієнтів С також буде поелементно відповідати посортованому списку коренів (повторювані значення у посортованому списку коренів стоять у порядку зростання степеня члена n, який стоїть біля них у загальному розв’язку).

-  matrix – функція, яка на ввід приймає список коефіцієнтів рекурсивного рівняння, а на вивід видає транзитивну матрицю цих коефіцієнтів у вигляді списку списків. Алгоритм: спочатку функція замінює знак останнього елементу на протилежний це пов’язано з тим, що останній елемент списку це an член і він знаходиться на іншій стороні рівняння (відділений від інших an-i членів), далі ми створюємо список у який будемо додавати списки (додані списки це рядки матриці) та опціональний аргумент який дорівнює -1 (ми його використаємо у наступних кроках), наступний крок ми запускаєм два цикли: перший цикл відповідає за кількість рядків, а другий - за кількість елементів у ряді, у кінці першого циклу до опціонального аргументу буде додаватися 1, далі якщо у другому циклі індекс елемента співпадатиме то у внутрішній список додається 1, якщо ні то 0, такий алгоритм відбувається для всіх стовпців матриці крім останнього, у останній стовпець (це останні елементи внутрішніх списків) ми перший елемент списку коефіцієнтів поділити на останній. Алгоритм продовжується поки матриця не буде заповнена.

- result – функція, яка знаходить n-ий розв’язок рекурентного рівняння. На ввід вона приймає 3 аргументи:  список коефіцієнтів рекурсивного рівняння; список, який містить початкові елементи рекурсивної послідовності; номер члена, який потрібно знайти, -  Алгоритм: за формулою, що an у нас дорівнює корені рівняння у степені n помножити на коефіцієнти рекурсивного рівняння, підставляючи у формулу перші n членів ми отримуємо розв’язки перших n  роз в’язків рекурентного рівняння.

- get_nth_el – на ввід приймає список перших членів рекурентної послідовності; матрицю (створено у функції matrix для заданих аргументів); число, яке вказує до якого число потрібно піднести матрицю.

 Для виконання даного проєкту ми використали знання набуті на заняттях з дискретної математики: розвязок рекурентних рівнянь, множення матриць; на парах з оп та деяку інформацію ми взяли з інтернету: метод Крамара, метод використаний у функції matrix та get_nth_el. Під час виконанння ми дізнались багато нового та отримали досвід, який пригодиться нам у майбутньому.

