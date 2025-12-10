"""
Architecture and Design Tests
Tests the architectural properties without requiring CuPy.
"""
import unittest
import sys
import os
import ast

class TestArchitecture(unittest.TestCase):
    """Test architectural design and code organization."""

    def test_project_structure(self):
        """Test that project has proper structure."""
        base_path = os.path.join(os.path.dirname(__file__), '..')

        # Check main directories exist
        self.assertTrue(os.path.exists(os.path.join(base_path, 'src')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'tests')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'src', 'engine')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'src', 'mcts')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'src', 'utils')))

    def test_module_separation(self):
        """Test that modules are properly separated."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')

        # Check __init__ files exist
        self.assertTrue(os.path.exists(os.path.join(src_path, '__init__.py')))
        self.assertTrue(os.path.exists(os.path.join(src_path, 'engine', '__init__.py')))
        self.assertTrue(os.path.exists(os.path.join(src_path, 'mcts', '__init__.py')))
        self.assertTrue(os.path.exists(os.path.join(src_path, 'utils', '__init__.py')))

    def test_game_state_class_exists(self):
        """Test that ChessState class has proper structure."""
        file_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'engine', 'game_state.py'
        )

        with open(file_path, 'r') as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Find ChessState class
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class_names = [cls.name for cls in classes]

        self.assertIn('ChessState', class_names)
        self.assertIn('GameException', class_names)

    def test_engine_class_exists(self):
        """Test that Chess5DEngine class exists."""
        file_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'engine', 'chess_engine.py'
        )

        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class_names = [cls.name for cls in classes]

        self.assertIn('Chess5DEngine', class_names)

    def test_mcts_classes_exist(self):
        """Test that MCTS classes exist."""
        node_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'mcts', 'node.py'
        )
        search_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'mcts', 'search.py'
        )

        with open(node_path, 'r') as f:
            node_content = f.read()
        with open(search_path, 'r') as f:
            search_content = f.read()

        node_tree = ast.parse(node_content)
        search_tree = ast.parse(search_content)

        node_classes = [n.name for n in node_tree.body if isinstance(n, ast.ClassDef)]
        search_classes = [n.name for n in search_tree.body if isinstance(n, ast.ClassDef)]

        self.assertIn('MCTSNode', node_classes)
        self.assertIn('MCTS', search_classes)

    def test_docstrings_present(self):
        """Test that modules have docstrings."""
        files_to_check = [
            'src/engine/game_state.py',
            'src/engine/chess_engine.py',
            'src/mcts/node.py',
            'src/mcts/search.py',
            'src/utils/js_interface.py'
        ]

        base_path = os.path.join(os.path.dirname(__file__), '..')

        for file_path in files_to_check:
            full_path = os.path.join(base_path, file_path)
            with open(full_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content)
            # Module should have docstring
            self.assertIsNotNone(ast.get_docstring(tree), f"{file_path} missing docstring")

    def test_code_organization(self):
        """Test that code is organized properly."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')

        # Engine module
        engine_files = os.listdir(os.path.join(src_path, 'engine'))
        self.assertIn('game_state.py', engine_files)
        self.assertIn('chess_engine.py', engine_files)

        # MCTS module
        mcts_files = os.listdir(os.path.join(src_path, 'mcts'))
        self.assertIn('node.py', mcts_files)
        self.assertIn('search.py', mcts_files)

        # Utils module
        utils_files = os.listdir(os.path.join(src_path, 'utils'))
        self.assertIn('js_interface.py', utils_files)


if __name__ == '__main__':
    unittest.main()
