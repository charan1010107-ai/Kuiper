import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter
    import lightgbm as lgb
    from sentence_transformers import SentenceTransformer
    HAS_HEAVY_ML = True
except ImportError:
    HAS_HEAVY_ML = False
    np = None
    SentenceTransformer = None

import query_handler
import general_knowledge
from tinyml import predict as tinyml_predict
from minhash import MinHashCache
from embedder import Embedder
from dynamic_alpha import get_alpha
from llm_client import call_llm

# ── COMPREHENSIVE MULTI-DOMAIN SURROGATE KNOWLEDGE BANK ────────
CATEGORY_RESPONSES = {
    # ── 1. CODING BUGS & SOFTWARE ENGINEERING (LAYER 4 SURROGATE) ──
    "python_closure_bug": (
        "🐛 **Python Late-Binding Closure Bug Analysis:**\n\n"
        "**Root Cause:** In Python, closures bind variables by *reference* (name lookup), not by value at creation time. "
        "When the lambdas execute in the loop, they all evaluate `i` at its final value (`4`), resulting in `[4*2, 4*2, 4*2, 4*2, 4*2] = [8, 8, 8, 8, 8]`.\n\n"
        "**Solution (Default Argument Capture):**\n"
        "Bind `i` as a default argument in the lambda definition so its current value is evaluated and stored at function creation:\n"
        "```python\ndef make_multipliers():\n    return [lambda x, i=i: i * x for i in range(5)]\n\nfor multiplier in make_multipliers():\n    print(multiplier(2))  # Output: 0, 2, 4, 6, 8\n```\n"
        "Alternatively, use `functools.partial(operator.mul, i)` for explicit functional binding."
    ),

    "python_mutable_default": (
        "🐛 **Python Mutable Default Argument Gotcha:**\n\n"
        "**Root Cause:** Default parameter values are evaluated *only once* when the function definition is executed, NOT each time the function is called. "
        "Consequently, the single list object `items=[]` persists across all subsequent function invocations.\n\n"
        "**Solution (Sentinel Pattern with None):**\n"
        "Use `None` as the default argument and instantiate a fresh list inside the function body:\n"
        "```python\ndef add_item(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items\n\nprint(add_item('apple'))   # ['apple']\nprint(add_item('banana'))  # ['banana'] (clean independent list)\n```"
    ),

    "js_loop_closure": (
        "🐛 **JavaScript Asynchronous Loop Scope Bug:**\n\n"
        "**Root Cause:** Variables declared with `var` are function-scoped, not block-scoped. By the time the `setTimeout` callbacks execute (after the event loop macrotask queue clears), the loop has already completed and `i` equals `3`.\n\n"
        "**Solution 1 (ES6 Block-Scoped `let` - Recommended):**\n"
        "```javascript\nfor (let i = 0; i < 3; i++) {\n    setTimeout(() => console.log(i), 100); // Outputs: 0, 1, 2\n}\n```\n\n"
        "**Solution 2 (IIFE Closure - ES5 Compatible):**\n"
        "```javascript\nfor (var i = 0; i < 3; i++) {\n    (function(currentI) {\n        setTimeout(() => console.log(currentI), 100);\n    })(i);\n}\n```"
    ),

    "bst_validation_bug": (
        "🐛 **Binary Search Tree (BST) Subtree Validation Bug:**\n\n"
        "**Root Cause:** Checking only immediate children (`node.left < node` and `node.right > node`) is insufficient. A valid BST requires *every* node in the left subtree to be strictly less than all ancestor roots, and every node in the right subtree to be strictly greater.\n\n"
        "**Correct Solution (Range-Bounded Recursion - O(N) Time, O(H) Space):**\n"
        "```python\ndef isValidBST(root):\n    def validate(node, low=float('-inf'), high=float('inf')):\n        if not node:\n            return True\n        if not (low < node.val < high):\n            return False\n        return validate(node.left, low, node.val) and validate(node.right, node.val, high)\n    \n    return validate(root)\n```"
    ),

    "react_state_mutation": (
        "⚛️ **React Direct State Mutation Bug:**\n\n"
        "**Root Cause:** Mutating state directly (e.g., `state.push(newItem)`) does not change the memory reference of the array/object. React's reconciliation engine uses shallow reference equality (`prev === next`) and fails to trigger a re-render.\n\n"
        "**Solution (Immutability with Spread Operator):**\n"
        "```javascript\n// ❌ Wrong: state.push(newItem); setState(state);\n// ✅ Correct:\nsetItems(prevItems => [...prevItems, newItem]);\n```"
    ),

    "cors_error_fix": (
        "🌐 **CORS (Cross-Origin Resource Sharing) Resolution:**\n\n"
        "**Root Cause:** The browser's Same-Origin Policy blocks frontend JavaScript from reading API responses from a different origin (domain, protocol, or port) unless the server explicitly sends CORS headers.\n\n"
        "**Server Fixes:**\n"
        "- **FastAPI (Python):**\n"
        "```python\nfrom fastapi.middleware.cors import CORSMiddleware\napp.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])\n```\n"
        "- **Express (Node.js):**\n"
        "```javascript\nconst cors = require('cors');\napp.use(cors());\n```"
    ),

    "sql_injection_fix": (
        "🔒 **SQL Injection Prevention:**\n\n"
        "**Root Cause:** Concatenating raw user inputs into SQL query strings allows malicious payload strings (e.g. `' OR 1=1 --`) to alter the SQL abstract syntax tree.\n\n"
        "**Solution (Parameterized / Prepared Queries):**\n"
        "```python\n# ❌ Vulnerable: cursor.execute(f'SELECT * FROM users WHERE email = \"{user_input}\"')\n# ✅ Secure Parameterized:\ncursor.execute('SELECT * FROM users WHERE email = %s', (user_input,))\n```"
    ),

    "n_plus_one_query": (
        "🗄️ **N+1 Database Query Problem & Optimization:**\n\n"
        "**Problem:** Fetching 1 parent record followed by N individual queries for each child relation (e.g., 100 users triggering 100 separate profile queries = 101 queries).\n\n"
        "**Resolution:** Use Eager Loading / Batch Joins:\n"
        "- **Django ORM:** `User.objects.select_related('profile')` or `prefetch_related('orders')`\n"
        "- **SQLAlchemy:** `session.query(User).options(joinedload(User.profile))`\n"
        "- **Raw SQL:** `SELECT * FROM users JOIN profiles ON users.id = profiles.user_id`"
    ),

    "big_o_complexity": (
        "📊 **Big-O Algorithm Complexity Hierarchy:**\n\n"
        "Ordered from fastest to slowest as $n \\to \\infty$:\n"
        "1. **$O(1)$ Constant:** Hash map lookup, array indexing\n"
        "2. **$O(\\log n)$ Logarithmic:** Binary search, balanced BST lookup\n"
        "3. **$O(n)$ Linear:** Single loop, linear search\n"
        "4. **$O(n \\log n)$ Linearithmic:** Merge Sort, QuickSort, TimSort\n"
        "5. **$O(n^2)$ Quadratic:** Nested loops, Bubble Sort, pairwise comparison\n"
        "6. **$O(2^n)$ Exponential:** Naive recursive Fibonacci, subset generation\n"
        "7. **$O(n!)$ Factorial:** Traveling Salesperson brute force, permutations"
    ),

    "p_vs_np_problem": (
        "🧠 **P vs NP Millennium Problem Overview:**\n\n"
        "- **P (Polynomial Time):** Problems that can be *solved* in polynomial time (e.g. shortest path, sorting, greatest common divisor).\n"
        "- **NP (Nondeterministic Polynomial Time):** Problems whose proposed solutions can be *verified* in polynomial time (e.g. Sudoku, Traveling Salesperson decision problem, Boolean Satisfiability).\n"
        "- **The Question ($P = NP$?):** If a problem's solution is easy to verify, is it necessarily easy to find? Most theoretical computer scientists conjecture $P \\neq NP$."
    ),

    # ── 2. CODE STRUCTURES & SKELETONS (LAYER 2/3/4 SURROGATE) ──
    "cpp_code_structure": (
        "💻 **Standard C++ Program Structure & Skeleton:**\n\n"
        "```cpp\n"
        "// 1. Documentation & Metadata Section\n"
        "/**\n"
        " * @file main.cpp\n"
        " * @brief Standard C++ Application Skeleton\n"
        " */\n\n"
        "// 2. Preprocessor Directives (Header File Inclusions)\n"
        "#include <iostream>   // For std::cin, std::cout, std::endl\n"
        "#include <vector>     // For standard sequence containers\n"
        "#include <string>     // For std::string string operations\n\n"
        "// 3. Namespace Declaration\n"
        "using namespace std;  // Brings std namespace symbols into global scope\n\n"
        "// 4. Global Constants, Macros, and Type Definitions\n"
        "constexpr int MAX_BUFFER_SIZE = 1024;\n\n"
        "// 5. Class / Struct Declarations\n"
        "class Calculator {\n"
        "private:\n"
        "    double result;\n"
        "public:\n"
        "    Calculator() : result(0.0) {}\n"
        "    double add(double a, double b) {\n"
        "        result = a + b;\n"
        "        return result;\n"
        "    }\n"
        "    double getResult() const { return result; }\n"
        "};\n\n"
        "// 6. Function Prototypes (Declarations)\n"
        "void greetUser(const string& username);\n\n"
        "// 7. Main Execution Function (Program Entry Point)\n"
        "int main(int argc, char* argv[]) {\n"
        "    // Local variable initialization\n"
        "    string name = \"Developer\";\n"
        "    greetUser(name);\n\n"
        "    // Class instantiation & method call\n"
        "    Calculator calc;\n"
        "    cout << \"Sum: \" << calc.add(10.5, 20.5) << endl;\n\n"
        "    // 8. Exit Code (0 signals successful execution to OS)\n"
        "    return 0;\n"
        "}\n\n"
        "// 9. Function Definitions (Implementations)\n"
        "void greetUser(const string& username) {\n"
        "    cout << \"Hello, \" << username << \"! Welcome to C++.\" << endl;\n"
        "}\n"
        "```\n\n"
        "**Core Architectural Components of a C++ Program:**\n"
        "1. **Preprocessor Directives (`#include`):** Instructs the compiler to include header files before compilation begins.\n"
        "2. **Namespace (`using namespace std`):** Prevents naming collisions and avoids prefixing standard utilities with `std::`.\n"
        "3. **Global Declarations / Classes:** Object-oriented definitions encapsulating attributes and methods.\n"
        "4. **Function Prototypes:** Declares function signature so `main()` can invoke functions defined later in the file.\n"
        "5. **`main()` Entry Point:** Every executable C++ program requires an `int main()` function that returns `0` upon success."
    ),

    "c_code_structure": (
        "💻 **Standard C Program Structure & Skeleton:**\n\n"
        "```c\n"
        "/* 1. Header Comments & Documentation */\n"
        "/* Filename: main.c */\n\n"
        "/* 2. Preprocessor Directives */\n"
        "#include <stdio.h>   /* Standard I/O: printf, scanf, puts */\n"
        "#include <stdlib.h>  /* Standard Library: malloc, free, exit */\n\n"
        "/* 3. Macros & Constants */\n"
        "#define SUCCESS 0\n"
        "#define MAX_LEN 100\n\n"
        "/* 4. Global Variables & Type Definitions (typedef struct) */\n"
        "typedef struct {\n"
        "    int id;\n"
        "    char name[MAX_LEN];\n"
        "} User;\n\n"
        "/* 5. Function Prototypes */\n"
        "void printUserInfo(const User* u);\n\n"
        "/* 6. Main Function (Entry Point) */\n"
        "int main(int argc, char *argv[]) {\n"
        "    User user1 = {1, \"Alice\"};\n"
        "    printUserInfo(&user1);\n"
        "    return SUCCESS;\n"
        "}\n\n"
        "/* 7. Function Definitions */\n"
        "void printUserInfo(const User* u) {\n"
        "    printf(\"User ID: %d | Name: %s\\n\", u->id, u->name);\n"
        "}\n"
        "```\n\n"
        "**Key Sections:** `#include` headers $\\to$ `#define` macros $\\to$ `struct/typedef` $\\to$ Function Prototypes $\\to$ `int main()` $\\to$ Function Definitions."
    ),

    "java_code_structure": (
        "☕ **Standard Java Class Structure & Skeleton:**\n\n"
        "```java\n"
        "// 1. Package Declaration (Must be the first line if present)\n"
        "package com.tokenwise.app;\n\n"
        "// 2. Import Statements\n"
        "import java.util.List;\n"
        "import java.util.ArrayList;\n\n"
        "// 3. Class Definition (Class name must match file name: Main.java)\n"
        "public class Main {\n"
        "    // 4. Class Constants and Static Variables\n"
        "    private static final String APP_NAME = \"TokenWise Java\";\n\n"
        "    // 5. Instance Fields\n"
        "    private int id;\n"
        "    private String name;\n\n"
        "    // 6. Constructor\n"
        "    public Main(int id, String name) {\n"
        "        this.id = id;\n"
        "        this.name = name;\n"
        "    }\n\n"
        "    // 7. Main Entry Point (JVM starts execution here)\n"
        "    public static void main(String[] args) {\n"
        "        System.out.println(\"Starting \" + APP_NAME + \"...\");\n"
        "        Main instance = new Main(101, \"Production\");\n"
        "        instance.display();\n"
        "    }\n\n"
        "    // 8. Instance Methods\n"
        "    public void display() {\n"
        "        System.out.println(\"Instance ID: \" + id + \", Name: \" + name);\n"
        "    }\n"
        "}\n"
        "```\n\n"
        "**Core Java Rules:**\n"
        "- Exactly one `public class` per `.java` file, matching the file name.\n"
        "- Execution begins unconditionally at `public static void main(String[] args)`."
    ),

    "python_code_structure": (
        "🐍 **Standard Python Script / Module Structure:**\n\n"
        "```python\n"
        "#!/usr/bin/env python3\n"
        "\"\"\"\n"
        "Module Docstring: High-level overview of the script's functionality.\n"
        "\"\"\"\n\n"
        "# 1. Standard Library Imports\n"
        "import os\n"
        "import sys\n"
        "from typing import List, Dict, Optional\n\n"
        "# 2. Third-Party Imports\n"
        "# import requests\n\n"
        "# 3. Local / Custom Package Imports\n"
        "# from core import utils\n\n"
        "# 4. Global Constants (UPPERCASE_WITH_UNDERSCORES)\n"
        "DEFAULT_TIMEOUT = 30\n"
        "MAX_RETRIES = 3\n\n"
        "# 5. Custom Classes\n"
        "class DataProcessor:\n"
        "    def __init__(self, data_source: str):\n"
        "        self.data_source = data_source\n\n"
        "    def process(self) -> Dict[str, int]:\n"
        "        return {\"status\": 200, \"items_processed\": 42}\n\n"
        "# 6. Helper Functions\n"
        "def setup_environment() -> None:\n"
        "    print(\"Initializing environment...\")\n\n"
        "# 7. Main Application Logic\n"
        "def main() -> int:\n"
        "    setup_environment()\n"
        "    processor = DataProcessor(\"source.csv\")\n"
        "    result = processor.process()\n"
        "    print(f\"Result: {result}\")\n"
        "    return 0\n\n"
        "# 8. Script Execution Guard (Prevents execution on import)\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
        "```"
    ),

    "rust_code_structure": (
        "🦀 **Standard Rust Program Structure & Skeleton:**\n\n"
        "```rust\n"
        "// 1. Module Imports and Crate Imports\n"
        "use std::error::Error;\n"
        "use std::fmt;\n\n"
        "// 2. Constants and Statics\n"
        "const APP_VERSION: &str = \"1.0.0\";\n\n"
        "// 3. Struct & Enum Type Definitions with Derive Macros\n"
        "#[derive(Debug, Clone)]\n"
        "pub struct Config {\n"
        "    pub port: u16,\n"
        "    pub host: String,\n"
        "}\n\n"
        "// 4. Implementation Blocks (Methods & Associated Functions)\n"
        "impl Config {\n"
        "    pub fn new(port: u16, host: &str) -> Self {\n"
        "        Config { port, host: host.to_string() }\n"
        "    }\n"
        "}\n\n"
        "// 5. Main Entry Point (Returns Result for idiomatic error handling)\n"
        "fn main() -> Result<(), Box<dyn Error>> {\n"
        "    println!(\"Starting Rust App v{}\", APP_VERSION);\n"
        "    let cfg = Config::new(8080, \"127.0.0.1\");\n"
        "    println!(\"Loaded Config: {:?}\", cfg);\n"
        "    Ok(())\n"
        "}\n"
        "```"
    ),

    "go_code_structure": (
        "🐹 **Standard Go (Golang) Program Structure:**\n\n"
        "```go\n"
        "// 1. Package Declaration (main for standalone executables)\n"
        "package main\n\n"
        "// 2. Grouped Imports\n"
        "import (\n"
        "    \"fmt\"\n"
        "    \"os\"\n"
        "    \"time\"\n"
        ")\n\n"
        "// 3. Constants and Package Variables\n"
        "const Version = \"1.0.0\"\n\n"
        "// 4. Struct and Interface Definitions\n"
        "type Server struct {\n"
        "    Host string\n"
        "    Port int\n"
        "}\n\n"
        "// 5. Receiver Methods\n"
        "func (s *Server) Start() {\n"
        "    fmt.Printf(\"Server listening on %s:%d\\n\", s.Host, s.Port)\n"
        "}\n\n"
        "// 6. Main Entry Point\n"
        "func main() {\n"
        "    fmt.Printf(\"App Version: %s (Started at %s)\\n\", Version, time.Now().Format(time.RFC3339))\n"
        "    srv := &Server{Host: \"localhost\", Port: 8080}\n"
        "    srv.Start()\n"
        "    os.Exit(0)\n"
        "}\n"
        "```"
    ),

    "html_code_structure": (
        "🌐 **Standard HTML5 Document Boilerplate & Structure:**\n\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "    <meta charset=\"UTF-8\">\n"
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "    <meta http-equiv=\"X-UA-Compatible\" content=\"ie=edge\">\n"
        "    <title>TokenWise Application</title>\n"
        "    <link rel=\"stylesheet\" href=\"style.css\">\n"
        "</head>\n"
        "<body>\n"
        "    <!-- Semantic Header & Navigation -->\n"
        "    <header>\n"
        "        <nav>\n"
        "            <h1>TokenWise</h1>\n"
        "        </nav>\n"
        "    </header>\n\n"
        "    <!-- Main Page Content -->\n"
        "    <main>\n"
        "        <section class=\"hero\">\n"
        "            <h2>Welcome to Intelligent Routing</h2>\n"
        "            <p>Optimizing latency and AI token spend.</p>\n"
        "        </section>\n"
        "    </main>\n\n"
        "    <!-- Semantic Footer -->\n"
        "    <footer>\n"
        "        <p>&copy; 2026 TokenWise. All rights reserved.</p>\n"
        "    </footer>\n\n"
        "    <!-- Scripts placed before closing body tag -->\n"
        "    <script src=\"app.js\"></script>\n"
        "</body>\n"
        "</html>\n"
        "```"
    ),

    "react_component_structure": (
        "⚛️ **Standard React Functional Component Structure:**\n\n"
        "```jsx\n"
        "// 1. Imports (React, Hooks, Child Components, Styles)\n"
        "import React, { useState, useEffect, useCallback } from 'react';\n"
        "import './UserProfile.css';\n\n"
        "// 2. Component Declaration with Props & Defaults\n"
        "const UserProfile = ({ userId, onUpdate, theme = 'dark' }) => {\n"
        "    // 3. State Hooks\n"
        "    const [user, setUser] = useState(null);\n"
        "    const [loading, setLoading] = useState(true);\n\n"
        "    // 4. Side Effects (Data fetching / lifecycle)\n"
        "    useEffect(() => {\n"
        "        let isMounted = true;\n"
        "        fetch(`/api/users/${userId}`)\n"
        "            .then(res => res.json())\n"
        "            .then(data => {\n"
        "                if (isMounted) {\n"
        "                    setUser(data);\n"
        "                    setLoading(false);\n"
        "                }\n"
        "            });\n"
        "        return () => { isMounted = false; };\n"
        "    }, [userId]);\n\n"
        "    // 5. Event Handlers\n"
        "    const handleRefresh = useCallback(() => {\n"
        "        setLoading(true);\n"
        "        if (onUpdate) onUpdate(userId);\n"
        "    }, [userId, onUpdate]);\n\n"
        "    // 6. Conditional Rendering Guard\n"
        "    if (loading) return <div className=\"spinner\">Loading profile...</div>;\n\n"
        "    // 7. JSX Template Return\n"
        "    return (\n"
        "        <div className={`profile-card ${theme}`}>\n"
        "            <h3>{user.name}</h3>\n"
        "            <p>Email: {user.email}</p>\n"
        "            <button onClick={handleRefresh} className=\"btn-refresh\">Refresh</button>\n"
        "        </div>\n"
        "    );\n"
        "};\n\n"
        "// 8. Default Export\n"
        "export default UserProfile;\n"
        "```"
    ),

    "sql_query_structure": (
        "🗄️ **Standard SQL Query Structure & Execution Order:**\n\n"
        "```sql\n"
        "-- 1. Written Order vs 2. Execution Order\n"
        "SELECT \n"
        "    department_id,\n"
        "    COUNT(employee_id) AS total_employees,\n"
        "    AVG(salary) AS avg_salary\n"
        "FROM employees\n"
        "INNER JOIN departments ON employees.dept_id = departments.id\n"
        "WHERE status = 'ACTIVE' AND hire_date >= '2020-01-01'\n"
        "GROUP BY department_id\n"
        "HAVING COUNT(employee_id) >= 5\n"
        "ORDER BY avg_salary DESC\n"
        "LIMIT 10 OFFSET 0;\n"
        "```\n\n"
        "**SQL Query Engine Order of Execution:**\n"
        "1. `FROM` & `JOIN` (Identify & join source tables)\n"
        "2. `WHERE` (Filter individual rows)\n"
        "3. `GROUP BY` (Aggregate rows into groups)\n"
        "4. `HAVING` (Filter aggregated groups)\n"
        "5. `SELECT` (Compute expressions and aliases)\n"
        "6. `DISTINCT` (Remove duplicate rows)\n"
        "7. `ORDER BY` (Sort final output rows)\n"
        "8. `LIMIT` / `OFFSET` (Paginate result set)"
    ),

    # ── 3. CUSTOMER SUPPORT DOMAINS ──
    "shipping_query":  "📦 [Order & Shipping] Your order tracking request has been processed. You can check real-time courier updates in your delivery status panel.",
    "refund_request":  "💳 [Refund & Returns] Your return/refund request has been initiated. Our policy allows returns within 30 days of delivery.",
    "payment_issue":   "🔒 [Billing & Payment] We've detected a payment inquiry. Your billing records are being verified with our payment gateway.",
    "cancellation":    "❌ [Subscription Management] We have processed your cancellation request. Any remaining active cycle will conclude at the end of the current billing period.",
    "address_update":  "📍 [Address Update] Your shipping address modification request has been queued for our logistics team.",
    "general_inquiry": "ℹ️ [Customer Support] Thank you for reaching out. How can our team assist you further with your account?",
}

# ── EXPANDED MULTI-DOMAIN TRAINING DATASET ─────────────────────
TRAINING_DATA = [
    # ── Code Structures & Language Boilerplates ──
    ("give me the code structure of c++", "cpp_code_structure"),
    ("what is the code structure of c++", "cpp_code_structure"),
    ("c++ code structure", "cpp_code_structure"),
    ("cpp code structure", "cpp_code_structure"),
    ("c++ basic program structure", "cpp_code_structure"),
    ("c++ boilerplate template", "cpp_code_structure"),
    ("c++ hello world structure", "cpp_code_structure"),
    ("basic structure of a c++ program", "cpp_code_structure"),
    ("explain c++ code structure", "cpp_code_structure"),
    ("c++ skeleton template", "cpp_code_structure"),
    ("standard c++ program skeleton", "cpp_code_structure"),
    ("c++ syntax structure", "cpp_code_structure"),
    ("c++ main function template structure", "cpp_code_structure"),

    ("give me the code structure of c", "c_code_structure"),
    ("what is the code structure of c language", "c_code_structure"),
    ("c program basic structure", "c_code_structure"),
    ("c language code structure", "c_code_structure"),
    ("c boilerplate skeleton", "c_code_structure"),
    ("basic structure of a c program", "c_code_structure"),
    ("c language program structure", "c_code_structure"),

    ("give me the code structure of java", "java_code_structure"),
    ("what is the code structure of java", "java_code_structure"),
    ("java class structure", "java_code_structure"),
    ("java basic program structure", "java_code_structure"),
    ("java boilerplate template", "java_code_structure"),
    ("basic java program skeleton", "java_code_structure"),
    ("java main method class skeleton", "java_code_structure"),

    ("give me the code structure of python", "python_code_structure"),
    ("what is the code structure of python script", "python_code_structure"),
    ("python script boilerplate template", "python_code_structure"),
    ("standard python program structure", "python_code_structure"),
    ("python main function structure", "python_code_structure"),
    ("basic structure of python code", "python_code_structure"),
    ("python module structure best practice", "python_code_structure"),

    ("give me the code structure of rust", "rust_code_structure"),
    ("what is the code structure of rust", "rust_code_structure"),
    ("rust program boilerplate template", "rust_code_structure"),
    ("rust basic code structure", "rust_code_structure"),
    ("rust main function struct skeleton", "rust_code_structure"),

    ("give me the code structure of go", "go_code_structure"),
    ("golang code structure", "go_code_structure"),
    ("go basic program boilerplate", "go_code_structure"),
    ("what is the code structure of golang", "go_code_structure"),
    ("go package main func main skeleton", "go_code_structure"),

    ("give me the code structure of html", "html_code_structure"),
    ("html5 boilerplate template", "html_code_structure"),
    ("html document basic structure", "html_code_structure"),
    ("what is the code structure of html5", "html_code_structure"),
    ("basic html web page structure", "html_code_structure"),

    ("give me the code structure of react component", "react_component_structure"),
    ("react functional component boilerplate template", "react_component_structure"),
    ("react component structure", "react_component_structure"),
    ("basic react component skeleton", "react_component_structure"),

    ("give me the structure of a sql query", "sql_query_structure"),
    ("sql query clause order structure", "sql_query_structure"),
    ("basic sql query syntax structure", "sql_query_structure"),
    ("order of sql clauses select from where group by", "sql_query_structure"),
    # ── Coding Bugs & Technical Problems ──
    ("why does this python code print 4 4 4 4 4 lambda in loop", "python_closure_bug"),
    ("python lambda inside list comprehension closure bug", "python_closure_bug"),
    ("make multipliers lambda return same value loop", "python_closure_bug"),
    ("python late binding closure in loop", "python_closure_bug"),
    ("fix python lambda multiplier closure", "python_closure_bug"),
    
    ("why does mutable default argument items list keep accumulating in python", "python_mutable_default"),
    ("python default argument items list bug add_item", "python_mutable_default"),
    ("mutable default argument unexpected behavior in python function", "python_mutable_default"),
    ("explain why add_item multiple times appends to same list", "python_mutable_default"),
    ("python function default list retains values across calls", "python_mutable_default"),

    ("why does javascript setTimeout in for loop output 3 3 3 instead of 0 1 2", "js_loop_closure"),
    ("javascript var in loop settimeout closure problem", "js_loop_closure"),
    ("fix javascript for loop settimeout printing same number", "js_loop_closure"),
    ("difference between var and let in settimeout loop", "js_loop_closure"),

    ("find the bug in recursive binary search tree isValidBST function", "bst_validation_bug"),
    ("isValidBST only checking immediate children bug", "bst_validation_bug"),
    ("how to properly validate a binary search tree in python", "bst_validation_bug"),
    ("valid binary search tree check subtree min max range", "bst_validation_bug"),

    ("why does mutating react state directly not trigger a re-render", "react_state_mutation"),
    ("react state push array does not re render component", "react_state_mutation"),
    ("react state immutability spread operator fix", "react_state_mutation"),

    ("how to fix CORS Access Control Allow Origin error in web app", "cors_error_fix"),
    ("cross origin request blocked CORS header missing", "cors_error_fix"),
    ("enable CORS in fastapi backend express nodejs", "cors_error_fix"),

    ("how to prevent SQL injection in database queries", "sql_injection_fix"),
    ("sql injection parameterized query vs string concatenation", "sql_injection_fix"),
    ("secure database query against sql injection attacks", "sql_injection_fix"),

    ("what is the N+1 query problem and how do i solve it", "n_plus_one_query"),
    ("orm N+1 queries optimization select_related prefetch_related", "n_plus_one_query"),
    ("database N+1 query problem eager loading join", "n_plus_one_query"),

    ("what is the Big O time complexity hierarchy from fastest to slowest", "big_o_complexity"),
    ("order of time complexity big O notations", "big_o_complexity"),
    ("compare O(1) O(log n) O(n) O(n log n) O(n^2)", "big_o_complexity"),

    ("explain the P vs NP problem in computer science in simple terms", "p_vs_np_problem"),
    ("what is the difference between P and NP complexity classes", "p_vs_np_problem"),
    ("is P equal to NP millennium prize problem", "p_vs_np_problem"),

    # ── Customer Support Domains ──
    ("where is my order", "shipping_query"),
    ("track my package please", "shipping_query"),
    ("i haven't received my delivery", "shipping_query"),
    ("when will my order arrive", "shipping_query"),
    ("has my order been dispatched", "shipping_query"),
    ("where is my package", "shipping_query"),
    ("what happened to my shipment", "shipping_query"),
    ("my parcel hasn't arrived", "shipping_query"),

    ("i want a refund", "refund_request"),
    ("give me my money back", "refund_request"),
    ("i need to return this item", "refund_request"),
    ("can i get a refund please", "refund_request"),
    ("please reimburse me", "refund_request"),
    ("refund my money", "refund_request"),

    ("my payment failed", "payment_issue"),
    ("i was billed two times", "payment_issue"),
    ("i was charged twice", "payment_issue"),
    ("billing problem on my card", "payment_issue"),
    ("wrong amount on my invoice", "payment_issue"),
    ("double charge on my account", "payment_issue"),

    ("cancel my subscription", "cancellation"),
    ("i want to unsubscribe", "cancellation"),
    ("please stop my plan", "cancellation"),
    ("terminate my account", "cancellation"),
    ("cancel my plan immediately", "cancellation"),

    ("change my delivery address", "address_update"),
    ("update my address please", "address_update"),
    ("new address for my order", "address_update"),
    ("i moved can you update my address", "address_update"),

    ("i need customer helpdesk", "general_inquiry"),
    ("how do i reset my account password", "general_inquiry"),
    ("what is your return and warranty policy", "general_inquiry"),
    ("talk to customer support representative", "general_inquiry"),
]

# ── ENSEMBLE MODEL (LAYER 4) ───────────────────────────────────
class EnsembleModel:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.le   = LabelEncoder()
        self.clf1 = LogisticRegression(max_iter=1000, C=2.0)
        self.clf2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42)
        self.clf3 = lgb.LGBMClassifier(n_estimators=120, random_state=42, verbose=-1)
        self._train()

    def _train(self):
        texts  = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]
        X      = self.embed_model.encode(texts)
        y      = self.le.fit_transform(labels)
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)
        self.clf3.fit(X, y)

    def predict(self, vec, alpha_threshold=0.30):
        v     = vec.reshape(1, -1)
        pred1 = self.le.inverse_transform(self.clf1.predict(v))[0]
        pred2 = self.le.inverse_transform(self.clf2.predict(v))[0]
        pred3 = self.le.inverse_transform(self.clf3.predict(v))[0]

        votes = [pred1, pred2, pred3]
        top_label, top_count = Counter(votes).most_common(1)[0]
        
        # LogReg calibrated probability confidence
        proba = self.clf1.predict_proba(v)[0]
        confidence = float(np.max(proba))
        
        # Effective threshold combines dynamic alpha with a strict domain floor
        effective_threshold = max(alpha_threshold, 0.62)
        
        # Agreement criteria: models agree AND confidence clears effective threshold AND response exists
        if top_count >= 2 and confidence >= effective_threshold and top_label in CATEGORY_RESPONSES:
            return {
                "label": top_label,
                "confidence": round(confidence, 3),
                "votes": votes,
                "pass_on": False
            }
        return {
            "label": None,
            "confidence": round(confidence, 3),
            "votes": votes,
            "pass_on": True
        }

# ── KUIPER ROUTER ──────────────────────────────────────────────
class KuiperRouter:
    def __init__(self):
        print("🚀 Initialising Kuiper Router Pipeline...")
        self.embed_model = None
        self.ensemble = None
        
        if HAS_HEAVY_ML and SentenceTransformer is not None:
            try:
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("  ✅ Layer 3 & 4 Embedding Model ready")
                self.ensemble = EnsembleModel(self.embed_model)
                print("  ✅ Layer 4 Ensemble (LogReg + MLP + LightGBM) ready")
            except Exception as e:
                print(f"  ⚠️ Skipping heavy ML initialization: {e}")
                self.embed_model = None
                self.ensemble = None
        else:
            print("  ℹ️ Running in lightweight serverless mode (Layers 0A, 0B, 1, 2, 5 fully active)")
        
        self.cache = MinHashCache()
        self._seed_cache()
        print("  ✅ Layer 2 MinHash LSH Cache ready")
        
        self.embedder = Embedder(model=self.embed_model)
        self.labeled_embeddings = self._build_labeled_embeddings() if (self.embedder and self.embedder.model) else []
        print("  ✅ Layer 3 Embedder ready")
        
        print("\n✨ Kuiper Router Pipeline fully active!\n")

    def _seed_cache(self):
        for text, label in TRAINING_DATA:
            self.cache.add(text, label)

    def _build_labeled_embeddings(self):
        store = []
        for text, label in TRAINING_DATA:
            vec = self.embedder.embed(text)
            store.append({"text": text, "label": label, "vector": vec})
        return store

    def add_learned_trace(self, query, label_or_answer):
        """Dynamic continuous learning from LLM traces"""
        try:
            self.cache.add(query, label_or_answer)
            if self.embedder and self.embedder.model:
                vec = self.embedder.embed(query)
                if vec is not None:
                    self.labeled_embeddings.append({"text": query, "label": label_or_answer, "vector": vec})
        except Exception as e:
            print(f"Error adding trace to memory: {e}")

    def route(self, query, provider="auto", api_key=None):
        """
        Execute exact 7-layer architecture pipeline:
        0A -> 0B -> 1 -> 2 -> 3 -> 4 -> 5
        """
        alpha_info = get_alpha()
        current_alpha = alpha_info["alpha"]
        steps_executed = []

        # ── LAYER 0A: Query Handler (Math, Greetings, Unit conversions, Facts) ──
        t0 = time.time()
        res0a = query_handler.handle(query)
        latency_0a = round((time.time() - t0) * 1000, 2)
        if not res0a["pass_on"]:
            steps_executed.append({"layer": "0A", "name": "Query Handler", "status": "HIT", "latency_ms": latency_0a})
            return {
                "answer": res0a["answer"],
                "handled_by": res0a["handled_by"],
                "layer": "0A",
                "layer_name": "Layer 0A: Query Handler",
                "cost_saved": True,
                "confidence": 1.0,
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "steps": steps_executed
            }
        steps_executed.append({"layer": "0A", "name": "Query Handler", "status": "SKIP", "latency_ms": latency_0a})

        # ── LAYER 0B: General Knowledge (Wikipedia, DuckDuckGo) ──
        t0 = time.time()
        res0b = general_knowledge.handle(query)
        latency_0b = round((time.time() - t0) * 1000, 2)
        if not res0b["pass_on"]:
            steps_executed.append({"layer": "0B", "name": "General Knowledge", "status": "HIT", "latency_ms": latency_0b})
            return {
                "answer": res0b["answer"],
                "handled_by": res0b["handled_by"],
                "layer": "0B",
                "layer_name": "Layer 0B: General Knowledge",
                "cost_saved": True,
                "confidence": 1.0,
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "steps": steps_executed
            }
        steps_executed.append({"layer": "0B", "name": "General Knowledge", "status": "SKIP", "latency_ms": latency_0b})

        # ── LAYER 1: TinyML (Keyword matching & Domain scoring) ──
        t0 = time.time()
        res1 = tinyml_predict(query)
        latency_1 = round((time.time() - t0) * 1000, 2)
        if not res1["pass_on"]:
            steps_executed.append({"layer": "1", "name": "TinyML", "status": "HIT", "latency_ms": latency_1})
            label = res1["label"]
            answer = CATEGORY_RESPONSES.get(label, f"Handled by TinyML ({label})")
            return {
                "answer": answer,
                "handled_by": "tinyml",
                "layer": "1",
                "layer_name": "Layer 1: TinyML Classifier",
                "cost_saved": True,
                "confidence": res1.get("confidence", 0.85),
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "steps": steps_executed
            }
        steps_executed.append({"layer": "1", "name": "TinyML", "status": "SKIP", "latency_ms": latency_1})

        # ── LAYER 2: MinHash Cache (LSH near-duplicate search) ──
        t0 = time.time()
        res2 = self.cache.search(query)
        latency_2 = round((time.time() - t0) * 1000, 2)
        if res2:
            steps_executed.append({"layer": "2", "name": "MinHash Cache", "status": "HIT", "latency_ms": latency_2})
            label = res2["label"]
            answer = CATEGORY_RESPONSES.get(label, str(label))
            return {
                "answer": answer,
                "handled_by": "cache",
                "layer": "2",
                "layer_name": "Layer 2: MinHash LSH Cache",
                "cost_saved": True,
                "confidence": res2.get("similarity", 0.95),
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "steps": steps_executed
            }
        steps_executed.append({"layer": "2", "name": "MinHash Cache", "status": "SKIP", "latency_ms": latency_2})

        # ── LAYER 3: Embedder (SentenceTransformer Semantic Similarity) ──
        t0 = time.time()
        query_vec = None
        if self.embedder and self.embedder.model and self.labeled_embeddings:
            try:
                query_vec = self.embedder.embed(query)
                res3 = self.embedder.find_similar(query_vec, self.labeled_embeddings)
                latency_3 = round((time.time() - t0) * 1000, 2)
                if res3 and res3.get("similarity", 0) >= 0.72:
                    steps_executed.append({"layer": "3", "name": "Embedder", "status": "HIT", "latency_ms": latency_3})
                    label = res3["label"]
                    answer = CATEGORY_RESPONSES.get(label, str(label))
                    return {
                        "answer": answer,
                        "handled_by": "embedder",
                        "layer": "3",
                        "layer_name": "Layer 3: Semantic Embedder",
                        "cost_saved": True,
                        "confidence": res3.get("similarity", 0.8),
                        "alpha": current_alpha,
                        "price_tier": alpha_info["tier"],
                        "steps": steps_executed
                    }
                steps_executed.append({"layer": "3", "name": "Embedder", "status": "SKIP", "latency_ms": latency_3})
            except Exception:
                steps_executed.append({"layer": "3", "name": "Embedder", "status": "SKIP", "latency_ms": 0.1})
        else:
            steps_executed.append({"layer": "3", "name": "Embedder", "status": "SKIP", "latency_ms": 0.1})

        # ── LAYER 4: Ensemble ML (LogReg + LightGBM + MLP with Dynamic Alpha) ──
        t0 = time.time()
        if self.ensemble and query_vec is not None:
            try:
                res4 = self.ensemble.predict(query_vec, alpha_threshold=current_alpha)
                latency_4 = round((time.time() - t0) * 1000, 2)
                if not res4["pass_on"]:
                    steps_executed.append({"layer": "4", "name": "Ensemble ML", "status": "HIT", "latency_ms": latency_4})
                    label = res4["label"]
                    answer = CATEGORY_RESPONSES.get(label, f"Classified by Ensemble ML ({label})")
                    return {
                        "answer": answer,
                        "handled_by": "ensemble",
                        "layer": "4",
                        "layer_name": "Layer 4: Ensemble ML (α-Calibrated)",
                        "cost_saved": True,
                        "confidence": res4.get("confidence", 0.75),
                        "alpha": current_alpha,
                        "price_tier": alpha_info["tier"],
                        "steps": steps_executed
                    }
                steps_executed.append({"layer": "4", "name": "Ensemble ML", "status": "SKIP", "latency_ms": latency_4})
            except Exception:
                steps_executed.append({"layer": "4", "name": "Ensemble ML", "status": "SKIP", "latency_ms": 0.1})
        else:
            steps_executed.append({"layer": "4", "name": "Ensemble ML", "status": "SKIP", "latency_ms": 0.1})

        # ── LAYER 5: LLM (Automated Model Selection & Comprehensive Answers) ──
        t0 = time.time()
        steps_executed.append({"layer": "5", "name": "LLM Inference", "status": "INVOKED", "latency_ms": 0})
        
        try:
            llm_result = call_llm(query, provider=provider, api_key=api_key)
            latency_5 = llm_result["latency_ms"]
            steps_executed[-1]["latency_ms"] = latency_5
            
            # Continuous learning: cache the LLM response for future near-duplicate matches
            self.add_learned_trace(query, llm_result["answer"])

            return {
                "answer": llm_result["answer"],
                "handled_by": llm_result["provider"],
                "layer": "5",
                "layer_name": f"Layer 5: LLM ({llm_result['provider'].title()} - {llm_result['model']})",
                "cost_saved": False,
                "confidence": 1.0,
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "model": llm_result["model"],
                "provider": llm_result["provider"],
                "selection_reason": llm_result.get("selection_reason"),
                "tokens": llm_result.get("tokens", 0),
                "cost": llm_result.get("cost", 0.0001),
                "steps": steps_executed
            }
        except Exception as e:
            return {
                "answer": f"LLM Error: {str(e)}",
                "handled_by": "error",
                "layer": "5",
                "layer_name": "Layer 5: LLM (Error)",
                "cost_saved": False,
                "confidence": 0.0,
                "alpha": current_alpha,
                "price_tier": alpha_info["tier"],
                "steps": steps_executed
            }