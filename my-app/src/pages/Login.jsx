import { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "react-toastify";

export default function Login() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (loading) return;

    try {
      setLoading(true);

      const res = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/auth/login`,
        formData
      );

      if (res.data?.token) {
        localStorage.setItem("userInside", res.data.token);
      }

      if (res.data?.user) {
        localStorage.setItem(
          "currentuser",
          JSON.stringify(res.data.user)
        );
      }

      toast.success("Login successful! 🎉");

      setTimeout(() => {
        navigate("/home");
      }, 1200);
    } catch (err) {
      console.error("Login Error:", err);

      if (err.response) {
        switch (err.response.status) {
          case 400:
          case 401:
            toast.error(err.response.data.message);
            break;

          case 500:
            toast.error("Server error. Please try again later.");
            break;

          default:
            toast.error("Something went wrong.");
        }
      } else if (err.request) {
        toast.error("Unable to reach the server.");
      } else {
        toast.error("Unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl bg-slate-900/60 backdrop-blur-xl rounded-3xl shadow-2xl border border-slate-800 overflow-hidden animate-fadeIn">
        <div className="grid md:grid-cols-2 gap-0">
          {/* Left Section */}
          <div className="bg-slate-900/95 p-8 md:p-10 relative animate-slideInLeft">
            <div className="flex items-center gap-3 mb-8">
              <div className="flex items-center gap-2 text-white font-bold text-xl tracking-wide">
                <span className="text-teal-400 text-2xl">🏥</span>
                <span className="text-teal-400">
                  MediCare Assistant
                </span>
              </div>
            </div>

            <h1 className="text-4xl font-bold text-white mb-2">
              Welcome Back
            </h1>

            <p className="text-gray-400 mb-8">
              Login to access your medical assistant
            </p>

            <form onSubmit={handleSubmit} className="space-y-6">
              <input
                type="email"
                placeholder="Email"
                value={formData.email}
                onChange={(e) =>
                  handleInputChange("email", e.target.value)
                }
                className="w-full bg-transparent border-b-2 border-gray-600 text-white placeholder-gray-500 py-3 px-1 focus:border-teal-500 focus:outline-none transition-colors"
                required
              />

              <input
                type="password"
                placeholder="Password"
                value={formData.password}
                onChange={(e) =>
                  handleInputChange("password", e.target.value)
                }
                className="w-full bg-transparent border-b-2 border-gray-600 text-white placeholder-gray-500 py-3 px-1 focus:border-teal-500 focus:outline-none transition-colors"
                required
              />

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-teal-600 to-cyan-700 hover:from-teal-700 hover:to-cyan-800 text-white font-semibold py-4 rounded-full transition-all transform hover:scale-[1.02] shadow-lg shadow-teal-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Logging in..." : "Log In"}
              </button>

              <p className="text-center text-sm text-gray-400 mt-4">
                New user?{" "}
                <Link
                  to="/signup"
                  className="text-teal-400 hover:text-teal-300 font-medium"
                >
                  Sign up here
                </Link>
              </p>
            </form>
          </div>

          {/* Right Section */}
          <div className="hidden md:block bg-slate-800/50 min-h-[600px] relative">
            <img
              src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&h=600&fit=crop"
              alt="Medical Assistant AI"
              className="w-full h-full object-cover opacity-70"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src =
                  "https://placehold.co/800x600/14b8a6/white?text=MediCare+AI";
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}