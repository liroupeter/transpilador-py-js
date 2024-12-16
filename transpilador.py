import re

def tokenize(code):
    tokens = []
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),
        ('IDENT',    r'[a-zA-Z_]\w*'),
        ('ASSIGN',   r'='),
        ('END',      r';'),
        ('OP',       r'[+\-*/]'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('IF',       r'if'),
        ('ELSE',     r'else'),
        ('WHILE',    r'while'),
        ('FOR',      r'for'),
        ('DEF',      r'def'),
        ('RETURN',   r'return'),
        ('LOGIC',    r'and|or'),
        ('COMPOP',   r'[<>]=?|==|!='),
        ('COLON',    r':'),
        ('INDENT',   r'\n[ ]+'),
        ('NEWLINE',  r'\n'),
        ('SKIP',     r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]

    tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'IDENT' and value in {'if', 'else', 'while', 'for', 'def', 'return'}:
            kind = value.upper()
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Unexpected character: {value}')
        tokens.append((kind, value))
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def consume(self, expected_type=None):
        current = self.peek()
        if expected_type and current[0] != expected_type:
            raise SyntaxError(f'Expected {expected_type} but got {current[0]}')
        self.pos += 1
        return current

    def parse(self):
        ast = []
        while self.pos < len(self.tokens):
            ast.append(self.parse_statement())
        return ast

    def parse_statement(self):
        token = self.peek()
        if token[0] == 'DEF':
            return self.parse_function()
        elif token[0] == 'IF':
            return self.parse_if()
        elif token[0] == 'WHILE':
            return self.parse_while()
        elif token[0] == 'FOR':
            return self.parse_for()
        elif token[0] == 'IDENT':
            return self.parse_assignment()
        else:
            raise SyntaxError(f'Unknown statement: {token}')

    def parse_function(self):
        self.consume('DEF')
        name = self.consume('IDENT')[1]
        self.consume('LPAREN')
        params = []
        while self.peek()[0] != 'RPAREN':
            params.append(self.consume('IDENT')[1])
            if self.peek()[0] == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')
        self.consume('COLON')
        body = []
        while self.peek()[0] not in {'NEWLINE', None}:
            body.append(self.parse_statement())
        return ('function', name, params, body)

    def parse_if(self):
        self.consume('IF')
        condition = self.parse_expression()
        self.consume('COLON')
        body = []
        while self.peek()[0] not in {'ELSE', 'NEWLINE', None}:
            body.append(self.parse_statement())
        else_body = None
        if self.peek()[0] == 'ELSE':
            self.consume('ELSE')
            self.consume('COLON')
            else_body = []
            while self.peek()[0] not in {'NEWLINE', None}:
                else_body.append(self.parse_statement())
        return ('if', condition, body, else_body)

    def parse_while(self):
        self.consume('WHILE')
        condition = self.parse_expression()
        self.consume('COLON')
        body = []
        while self.peek()[0] != 'NEWLINE':
            body.append(self.parse_statement())
        return ('while', condition, body)

    def parse_for(self):
        self.consume('FOR')
        var = self.consume('IDENT')[1]
        self.consume('IN')
        iterable = self.parse_expression()
        self.consume('COLON')
        body = []
        while self.peek()[0] != 'NEWLINE':
            body.append(self.parse_statement())
        return ('for', var, iterable, body)

    def parse_assignment(self):
        var_name = self.consume('IDENT')[1]
        self.consume('ASSIGN')
        value = self.parse_expression()
        self.consume('END')
        return ('assign', var_name, value)

    def parse_expression(self):
        left = self.consume()[1]
        if self.peek()[0] == 'OP':
            op = self.consume('OP')[1]
            right = self.consume()[1]
            return ('binary_op', op, left, right)
        return left

class Transpiler:
    def transpile(self, ast):
        js_code = ''
        for node in ast:
            js_code += self.transpile_node(node) + '\n'
        return js_code

    def transpile_node(self, node):
        if node[0] == 'assign':
            return f'let {node[1]} = {self.transpile_expression(node[2])};'
        elif node[0] == 'binary_op':
            return f'({node[1]} {node[2]} {node[3]})'
        elif node[0] == 'function':
            params = ', '.join(node[2])
            body = '\n'.join(self.transpile_node(stmt) for stmt in node[3])
            return f'function {node[1]}({params}) {{\n{body}\n}}'
        elif node[0] == 'if':
            condition = self.transpile_expression(node[1])
            body = '\n'.join(self.transpile_node(stmt) for stmt in node[2])
            else_body = '\n'.join(self.transpile_node(stmt) for stmt in node[3]) if node[3] else ''
            return f'if ({condition}) {{\n{body}\n}} else {{\n{else_body}\n}}'
        elif node[0] == 'while':
            condition = self.transpile_expression(node[1])
            body = '\n'.join(self.transpile_node(stmt) for stmt in node[2])
            return f'while ({condition}) {{\n{body}\n}}'
        elif node[0] == 'for':
            iterable = self.transpile_expression(node[2])
            body = '\n'.join(self.transpile_node(stmt) for stmt in node[3])
            return f'for (let {node[1]} of {iterable}) {{\n{body}\n}}'

    def transpile_expression(self, expr):
        if isinstance(expr, tuple) and expr[0] == 'binary_op':
            return f'({expr[2]} {expr[1]} {expr[3]})'
        return str(expr)

# Exemplo de uso
code = """
def add(a, b):
    return a + b

x = 10;
if x > 5:
    x = x + 1;
"""

tokens = tokenize(code)
parser = Parser(tokens)
ast = parser.parse()
transpiler = Transpiler()
js_code = transpiler.transpile(ast)
print(js_code)
