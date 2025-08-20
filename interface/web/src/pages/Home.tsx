import React from 'react';
import { ArrowRight, CheckCircle, Star } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home = () => {
  const features = [
    'Custom Blueprint Design',
    '3D Visualization',
    'Renovation Planning',
    'Expert Consultation',
  ];

  const testimonials = [
    {
      name: 'Sarah Johnson',
      role: 'Homeowner',
      content: 'BlueprintPro turned our dream home into reality. The visualization tools were incredibly helpful in making decisions.',
      rating: 5,
    },
    {
      name: 'Michael Chen',
      role: 'Property Developer',
      content: "The team's expertise and attention to detail made our project a success. Highly recommended!",
      rating: 5,
    },
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div 
        className="relative h-[80vh] bg-cover bg-center"
        style={{
          backgroundImage: 'url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80")',
        }}
      >
        <div className="absolute inset-0 bg-black bg-opacity-50" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex items-center">
          <div className="text-white">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 animate-card">
              Design Your Dream Home
              <br />
              <span className="text-blue-400">Blueprint by Blueprint</span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 max-w-2xl animate-card animation-delay-100">
              Professional residential blueprint design services to help you visualize and plan your perfect home.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 animate-card animation-delay-200">
              <Link
                to="/design"
                className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-medium hover:bg-blue-700 transition-colors inline-flex items-center"
              >
                Start Designing
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link
                to="/portfolio"
                className="bg-white text-gray-900 px-8 py-4 rounded-md text-lg font-medium hover:bg-gray-100 transition-colors"
              >
                View Portfolio
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4 animate-card">
              Why Choose BlueprintPro?
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto animate-card animation-delay-100">
              We combine expertise with cutting-edge technology to bring your vision to life.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div 
                key={feature} 
                className="bg-white p-6 rounded-lg shadow-md animate-card"
                style={{ animationDelay: `${(index + 1) * 100}ms` }}
              >
                <CheckCircle className="h-12 w-12 text-blue-600 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature}</h3>
                <p className="text-gray-600">
                  Professional solutions tailored to your needs.
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-16 animate-card">What Our Clients Say</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {testimonials.map((testimonial, index) => (
              <div 
                key={index} 
                className="bg-white p-8 rounded-lg shadow-md animate-card"
                style={{ animationDelay: `${(index + 1) * 100}ms` }}
              >
                <div className="flex mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 mb-4">{testimonial.content}</p>
                <div>
                  <p className="font-semibold">{testimonial.name}</p>
                  <p className="text-gray-500">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;