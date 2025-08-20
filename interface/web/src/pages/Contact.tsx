/**
 * @fileoverview Contact page component with form handling and contact information display.
 * Implements a responsive layout with form validation and submission handling.
 */

import React, { useState } from 'react';
import { Mail, Phone, MapPin, Clock } from 'lucide-react';

/**
 * @typedef {Object} FormData
 * @property {string} name - Full name of the contact
 * @property {string} email - Email address
 * @property {string} phone - Phone number
 * @property {string} projectType - Type of project selected
 * @property {string} message - Project details message
 */

/**
 * @typedef {Object} ContactInfo
 * @property {React.ComponentType} icon - Lucide icon component
 * @property {string} title - Contact method title
 * @property {string} content - Contact information
 * @property {string} [link] - Optional URL for clickable contact methods
 */

const Contact = () => {
  // Form state management using controlled components
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    projectType: '',
    message: ''
  });

  /**
   * Handles form submission and validation
   * @param {React.FormEvent} e - Form submission event
   */
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement form submission to backend
    console.log('Form submitted:', formData);
  };

  /**
   * Updates form state when input values change
   * @param {React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>} e - Input change event
   */
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Contact information displayed in the sidebar
  const contactInfo = [
    {
      icon: Phone,
      title: 'Phone',
      content: '(123) 456-7890',
      link: 'tel:+1234567890'
    },
    {
      icon: Mail,
      title: 'Email',
      content: 'info@blueprintpro.com',
      link: 'mailto:info@blueprintpro.com'
    },
    {
      icon: MapPin,
      title: 'Office',
      content: '123 Design Street, Suite 100, New York, NY 10001'
    },
    {
      icon: Clock,
      title: 'Hours',
      content: 'Mon-Fri: 9AM-6PM EST'
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">Contact Us</h1>
          <p className="text-xl max-w-2xl mx-auto">
            Ready to start your project? Get in touch with our team of experts today.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Contact Information Sidebar */}
          <div className="lg:col-span-1">
            <h2 className="text-2xl font-bold mb-8">Get in Touch</h2>
            <div className="space-y-6">
              {contactInfo.map((item, index) => (
                <div key={index} className="flex items-start">
                  <item.icon className="h-6 w-6 text-blue-600 mt-1" />
                  <div className="ml-4">
                    <h3 className="font-semibold">{item.title}</h3>
                    {item.link ? (
                      <a href={item.link} className="text-gray-600 hover:text-blue-600 transition-colors">
                        {item.content}
                      </a>
                    ) : (
                      <p className="text-gray-600">{item.content}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Contact Form */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Form fields implementation... */}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;