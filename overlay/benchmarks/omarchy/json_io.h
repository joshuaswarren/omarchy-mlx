// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Tiny strict JSON reader/writer for the spike receipt and the pinned
// comparator file. No external dependency: the spike builds with Vulkan
// headers and the mlx library alone.
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace omarchy_spike::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

class Value {
 public:
  enum class Kind { Null, Bool, Number, String, ArrayT, ObjectT };

  Value() : kind_(Kind::Null) {}
  explicit Value(bool b) : kind_(Kind::Bool), bool_(b) {}
  explicit Value(double d) : kind_(Kind::Number), num_(d) {}
  explicit Value(const char* s) : kind_(Kind::String), str_(s) {}
  explicit Value(std::string s) : kind_(Kind::String), str_(std::move(s)) {}
  explicit Value(Array a)
      : kind_(Kind::ArrayT), arr_(std::make_shared<Array>(std::move(a))) {}
  explicit Value(Object o)
      : kind_(Kind::ObjectT), obj_(std::make_shared<Object>(std::move(o))) {}

  Kind kind() const {
    return kind_;
  }
  bool is_null() const {
    return kind_ == Kind::Null;
  }
  bool is_number() const {
    return kind_ == Kind::Number;
  }
  bool is_string() const {
    return kind_ == Kind::String;
  }
  bool is_object() const {
    return kind_ == Kind::ObjectT;
  }
  bool is_array() const {
    return kind_ == Kind::ArrayT;
  }

  double number() const {
    require(Kind::Number, "number");
    return num_;
  }
  const std::string& string() const {
    require(Kind::String, "string");
    return str_;
  }
  bool boolean() const {
    require(Kind::Bool, "bool");
    return bool_;
  }
  const Object& object() const {
    require(Kind::ObjectT, "object");
    return *obj_;
  }
  const Array& array() const {
    require(Kind::ArrayT, "array");
    return *arr_;
  }

  // Object member lookup; throws with a dotted-path message when absent.
  const Value& at(const std::string& key, const std::string& path = "") const {
    require(Kind::ObjectT, "object");
    auto it = obj_->find(key);
    if (it == obj_->end()) {
      throw std::runtime_error(
          "missing field '" + (path.empty() ? key : path + "." + key) + "'");
    }
    return it->second;
  }
  bool has(const std::string& key) const {
    return kind_ == Kind::ObjectT && obj_->count(key) > 0;
  }

  std::string dump(int indent = 2) const;
  void dump_to(std::ostream& os, int indent, int depth) const;

 private:
  void require(Kind k, const char* what) const {
    if (kind_ != k) {
      throw std::runtime_error(
          std::string("JSON type error: expected ") + what);
    }
  }

  Kind kind_;
  bool bool_{false};
  double num_{0};
  std::string str_;
  std::shared_ptr<Array> arr_;
  std::shared_ptr<Object> obj_;
};

inline void Value::dump_to(std::ostream& os, int indent, int depth) const {
  auto pad = [&](int d) {
    if (indent <= 0) {
      return;
    }
    os << "\n" << std::string(indent * d, ' ');
  };
  switch (kind_) {
    case Kind::Null:
      os << "null";
      break;
    case Kind::Bool:
      os << (bool_ ? "true" : "false");
      break;
    case Kind::Number: {
      if (std::isfinite(num_) && num_ == std::floor(num_) &&
          std::fabs(num_) < 1e15) {
        os << (long long)num_;
      } else {
        os.precision(10);
        os << num_;
      }
      break;
    }
    case Kind::String: {
      os << '"';
      for (unsigned char c : str_) {
        switch (c) {
          case '"':
            os << "\\\"";
            break;
          case '\\':
            os << "\\\\";
            break;
          case '\n':
            os << "\\n";
            break;
          case '\r':
            os << "\\r";
            break;
          case '\t':
            os << "\\t";
            break;
          default:
            if (c < 0x20) {
              char buf[8];
              snprintf(buf, sizeof(buf), "\\u%04x", c);
              os << buf;
            } else {
              os << c;
            }
        }
      }
      os << '"';
      break;
    }
    case Kind::ArrayT: {
      os << '[';
      bool first = true;
      for (auto& v : *arr_) {
        if (!first)
          os << ',';
        first = false;
        pad(depth + 1);
        v.dump_to(os, indent, depth + 1);
      }
      if (!arr_->empty())
        pad(depth);
      os << ']';
      break;
    }
    case Kind::ObjectT: {
      os << '{';
      bool first = true;
      for (auto& [k, v] : *obj_) {
        if (!first)
          os << ',';
        first = false;
        pad(depth + 1);
        Value(k).dump_to(os, 0, 0);
        os << (indent > 0 ? ": " : ":");
        v.dump_to(os, indent, depth + 1);
      }
      if (!obj_->empty())
        pad(depth);
      os << '}';
      break;
    }
  }
}

inline std::string Value::dump(int indent) const {
  std::ostringstream os;
  dump_to(os, indent, 0);
  return os.str();
}

// --- Parser ----------------------------------------------------------------

class Parser {
 public:
  explicit Parser(std::string text) : s_(std::move(text)) {}

  Value parse() {
    skip_ws();
    Value v = parse_value(0);
    skip_ws();
    if (pos_ != s_.size()) {
      fail("trailing content");
    }
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(
        "JSON parse error at offset " + std::to_string(pos_) + ": " + msg);
  }
  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' ||
            s_[pos_] == '\r')) {
      pos_++;
    }
  }
  char peek() {
    if (pos_ >= s_.size())
      fail("unexpected end");
    return s_[pos_];
  }
  void expect(char c) {
    if (pos_ >= s_.size() || s_[pos_] != c) {
      fail(std::string("expected '") + c + "'");
    }
    pos_++;
  }

  Value parse_value(int depth) {
    if (depth > 64)
      fail("nesting too deep");
    skip_ws();
    char c = peek();
    if (c == '{')
      return parse_object(depth);
    if (c == '[')
      return parse_array(depth);
    if (c == '"')
      return Value(parse_string());
    if (c == 't')
      return parse_lit("true", Value(true));
    if (c == 'f')
      return parse_lit("false", Value(false));
    if (c == 'n')
      return parse_lit("null", Value());
    return parse_number();
  }
  Value parse_lit(const std::string& lit, Value v) {
    if (s_.compare(pos_, lit.size(), lit) != 0)
      fail("bad literal");
    pos_ += lit.size();
    return v;
  }
  Value parse_object(int depth) {
    expect('{');
    Object o;
    skip_ws();
    if (peek() == '}') {
      pos_++;
      return Value(std::move(o));
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      o.emplace(key, parse_value(depth + 1));
      skip_ws();
      char c = peek();
      if (c == ',') {
        pos_++;
        continue;
      }
      if (c == '}') {
        pos_++;
        break;
      }
      fail("expected ',' or '}'");
    }
    return Value(std::move(o));
  }
  Value parse_array(int depth) {
    expect('[');
    Array a;
    skip_ws();
    if (peek() == ']') {
      pos_++;
      return Value(std::move(a));
    }
    while (true) {
      a.push_back(parse_value(depth + 1));
      skip_ws();
      char c = peek();
      if (c == ',') {
        pos_++;
        continue;
      }
      if (c == ']') {
        pos_++;
        break;
      }
      fail("expected ',' or ']'");
    }
    return Value(std::move(a));
  }
  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size())
        fail("unterminated string");
      char c = s_[pos_++];
      if (c == '"')
        break;
      if (c == '\\') {
        if (pos_ >= s_.size())
          fail("bad escape");
        char e = s_[pos_++];
        switch (e) {
          case '"':
            out += '"';
            break;
          case '\\':
            out += '\\';
            break;
          case '/':
            out += '/';
            break;
          case 'n':
            out += '\n';
            break;
          case 'r':
            out += '\r';
            break;
          case 't':
            out += '\t';
            break;
          case 'b':
            out += '\b';
            break;
          case 'f':
            out += '\f';
            break;
          case 'u': {
            if (pos_ + 4 > s_.size())
              fail("bad \\u");
            unsigned cp = std::stoul(s_.substr(pos_, 4), nullptr, 16);
            pos_ += 4;
            // Surrogate pairs map to UTF-8; BMP chars encode directly.
            if (cp >= 0xD800 && cp < 0xDC00 && pos_ + 6 <= s_.size() &&
                s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              unsigned lo = std::stoul(s_.substr(pos_ + 2, 4), nullptr, 16);
              if (lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                pos_ += 6;
              }
            }
            if (cp < 0x80) {
              out += char(cp);
            } else if (cp < 0x800) {
              out += char(0xC0 | (cp >> 6));
              out += char(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
              out += char(0xE0 | (cp >> 12));
              out += char(0x80 | ((cp >> 6) & 0x3F));
              out += char(0x80 | (cp & 0x3F));
            } else {
              out += char(0xF0 | (cp >> 18));
              out += char(0x80 | ((cp >> 12) & 0x3F));
              out += char(0x80 | ((cp >> 6) & 0x3F));
              out += char(0x80 | (cp & 0x3F));
            }
            break;
          }
          default:
            fail("bad escape char");
        }
      } else {
        out += c;
      }
    }
    return out;
  }
  Value parse_number() {
    size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+'))
      pos_++;
    bool has_digits = false;
    while (pos_ < s_.size() &&
           (isdigit((unsigned char)s_[pos_]) || s_[pos_] == '.' ||
            s_[pos_] == 'e' || s_[pos_] == 'E' || s_[pos_] == '+' ||
            s_[pos_] == '-')) {
      has_digits = has_digits || isdigit((unsigned char)s_[pos_]);
      pos_++;
    }
    if (!has_digits)
      fail("bad number");
    try {
      return Value(std::stod(s_.substr(start, pos_ - start)));
    } catch (...) {
      fail("bad number");
    }
  }

  std::string s_;
  size_t pos_{0};
};

inline Value parse(const std::string& text) {
  return Parser(text).parse();
}

// Convenience builders for receipts.
inline Value obj(Object o) {
  return Value(std::move(o));
}
inline Value arr(Array a) {
  return Value(std::move(a));
}

} // namespace omarchy_spike::json
